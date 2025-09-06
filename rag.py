"""
RAG core with auto-create Pinecone index (serverless) and robust error handling.
Compatible with the modern `pinecone` SDK (renamed from pinecone-client).
Behavior preserved; refactor improves docs, typing, and small safety checks.
"""
from __future__ import annotations

import logging
import os
import time
from functools import cached_property
from typing import Any, Optional, Protocol, Literal

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

__all__ = [
    "RAG",
    "RAGConfig",
    "RAGError",
    "IndexNotFoundError",
    "RetrievalError",
    "ensure_index",
]

# region: config (env can override; defaults to free-tier compatible us-east-1)
PINECONE_REGION = os.environ.get("PINECONE_REGION", "us-east-1")
# endregion


# --------- helpers ---------
def _safe_list_index_names(pc: Pinecone) -> list[str]:
    """Return index names (supports v5 `pinecone` SDK object or dict shapes)."""
    try:
        items = pc.list_indexes()  # iterable of objects with .name
        return [getattr(i, "name", None) or (i.get("name") if isinstance(i, dict) else None) for i in items]
    except Exception:
        # Some SDK variants expose .names()
        try:
            return pc.list_indexes().names()
        except Exception:
            return []


def _get_embedding_dimension(embedding: Embeddings) -> int:
    """Infer embedding dimension by embedding a short probe string."""
    vec = embedding.embed_query("dim-probe")
    return len(vec)


def ensure_index(
    pc: Pinecone,
    name: str,
    dim: int,
    *,
    metric: Literal["cosine", "dotproduct", "euclidean"] = "cosine",
    cloud: Literal["aws", "gcp", "azure"] = "aws",
    region: str = PINECONE_REGION,
    wait_ready: bool = True,
    timeout_s: int = 120,
) -> None:
    """
    Create a serverless index if missing and verify its dimension matches `dim`.

    Raises:
        RuntimeError: If an existing index has a mismatched dimension.
        TimeoutError: If waiting for readiness times out.
    """
    existing = _safe_list_index_names(pc)
    if name in existing:
        desc = pc.describe_index(name)
        idx_dim: Optional[int] = getattr(desc, "dimension", None)
        if idx_dim is None and isinstance(desc, dict):
            idx_dim = desc.get("dimension")
        if idx_dim is not None and idx_dim != dim:
            raise RuntimeError(
                f"Index '{name}' exists with dimension {idx_dim}, expected {dim}. "
                f"Use a new index name or recreate it to match."
            )
        return

    logger.info(f"Creating Pinecone index '%s' (dim=%s, metric=%s, %s/%s)...", name, dim, metric, cloud, region)
    pc.create_index(
        name=name,
        dimension=dim,
        metric=metric,
        spec=ServerlessSpec(cloud=cloud, region=region),
    )

    if not wait_ready:
        return

    deadline = time.time() + timeout_s
    while True:
        desc = pc.describe_index(name)
        status = getattr(desc, "status", None) or (desc.get("status") if isinstance(desc, dict) else {}) or {}
        if status.get("ready"):
            break
        if time.time() > deadline:
            raise TimeoutError(f"Timed out waiting for Pinecone index '{name}' to be ready.")
        time.sleep(2)


# --------- typing surface ---------
class VectorStoreProtocol(Protocol):
    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]: ...
    def similarity_search_with_score(self, query: str, k: int = 4, **kwargs: Any) -> list[tuple[Document, float]]: ...
    def as_retriever(self, **kwargs: Any): ...


class RAGConfig(BaseModel):
    """Runtime config for RAG behavior."""
    default_top_k: int = Field(default=3, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    enable_metadata_filtering: bool = Field(default=True)

    class Config:
        frozen = True


# --------- exceptions ---------
class RAGError(Exception): ...
class IndexNotFoundError(RAGError): ...
class RetrievalError(RAGError): ...


# --------- core ---------
class RAG:
    """
    Thin RAG wrapper around a Pinecone vector index and a LangChain Embeddings model.

    - Auto-creates a serverless Pinecone index if missing (dimension inferred from embeddings).
    - Provides similarity lookup APIs and a LangChain retriever.
    """

    def __init__(
        self,
        pinecone_client: Pinecone,
        pinecone_index_name: str,
        embedding: Embeddings,
        config: RAGConfig | None = None,
    ) -> None:
        self._pinecone_client = pinecone_client
        self._pinecone_index_name = pinecone_index_name
        self._embedding = embedding
        self._config = config or RAGConfig()

        if not self._pinecone_index_name:
            raise ValueError("Pinecone index name cannot be empty")
        if not self._pinecone_client:
            raise ValueError("Pinecone client is required")
        if not self._embedding:
            raise ValueError("Embedding model is required")

        # Auto-create (serverless) and validate dimension
        emb_dim = _get_embedding_dimension(self._embedding)
        ensure_index(self._pinecone_client, self._pinecone_index_name, emb_dim)

        self._validate_initialization()
        logger.info("RAG initialized with index '%s' and config: %s", pinecone_index_name, self._config)

    def _validate_initialization(self) -> None:
        """Verify the target index exists and is reachable."""
        try:
            available_indexes = _safe_list_index_names(self._pinecone_client)
            if self._pinecone_index_name not in available_indexes:
                raise IndexNotFoundError(
                    f"Index '{self._pinecone_index_name}' not found. Available indexes: {available_indexes}"
                )
        except Exception as e:
            logger.error("Failed to validate index existence: %s", e)
            raise IndexNotFoundError(f"Unable to verify index '{self._pinecone_index_name}': {e}") from e

    @cached_property
    def _vector_store(self) -> PineconeVectorStore:
        """Memoized LangChain vector store backed by the configured Pinecone index."""
        try:
            index = self._pinecone_client.Index(self._pinecone_index_name)
            return PineconeVectorStore(index=index, embedding=self._embedding)
        except Exception as e:
            logger.error("Failed to initialize vector store: %s", e)
            raise RAGError(f"Vector store initialization failed: {e}") from e

    # ----- lookups -----
    def lookup(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Basic similarity search.

        Raises:
            ValueError: Invalid query or top_k.
            RetrievalError: Underlying vector store failure.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace")
        k = int(top_k or self._config.default_top_k)
        if k <= 0:
            raise ValueError(f"top_k must be positive, got {k}")

        try:
            kwargs: dict[str, Any] = {"k": k}
            if filter_metadata and self._config.enable_metadata_filtering:
                kwargs["filter"] = filter_metadata
            return self._vector_store.similarity_search(query, **kwargs)
        except Exception as e:
            logger.error("Document retrieval failed: %s", e)
            raise RetrievalError(f"Failed to retrieve documents: {e}") from e

    def lookup_with_scores(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Similarity search with scores.

        Note: Depending on vector store settings, the score may represent similarity
        (higher is better) or distance (lower is better). This class leaves semantics
        unchanged and applies an optional threshold as configured by `similarity_threshold`.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty or whitespace")
        k = int(top_k or self._config.default_top_k)

        try:
            kwargs: dict[str, Any] = {"k": k}
            if filter_metadata and self._config.enable_metadata_filtering:
                kwargs["filter"] = filter_metadata

            results = self._vector_store.similarity_search_with_score(query, **kwargs)

            # Keep original semantics: filter if threshold > 0.
            # If your backend returns distance, consider adapting this comparison.
            thr = self._config.similarity_threshold
            if thr > 0:
                results = [(d, s) for d, s in results if s >= thr]
            return results
        except Exception as e:
            logger.error("Document retrieval with scores failed: %s", e)
            raise RetrievalError(f"Failed to retrieve documents with scores: {e}") from e

    # ----- retriever -----
    def get_retriever(self, **kwargs: Any):
        """
        Return a LangChain retriever. Defaults to similarity search with k from config.
        `kwargs` may include `search_type` and `search_kwargs`.
        """
        try:
            search_kwargs = dict(kwargs.get("search_kwargs", {}))
            search_kwargs.setdefault("k", self._config.default_top_k)

            retriever_kwargs = {
                "search_type": kwargs.get("search_type", "similarity"),
                "search_kwargs": search_kwargs,
            }
            return self._vector_store.as_retriever(**retriever_kwargs)
        except Exception as e:
            logger.error("Failed to create retriever: %s", e)
            raise RAGError(f"Retriever creation failed: {e}") from e

    # ----- stats -----
    def get_index_stats(self) -> dict[str, Any]:
        """Return basic index and config stats."""
        try:
            index = self._pinecone_client.Index(self._pinecone_index_name)
            stats = index.describe_index_stats()
            desc = self._pinecone_client.describe_index(self._pinecone_index_name)
            dimension = getattr(desc, "dimension", None)
            if dimension is None and isinstance(desc, dict):
                dimension = desc.get("dimension")

            # pydantic v1/v2 compatible dump
            cfg = self._config
            config_dump = cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()

            return {
                "index_name": self._pinecone_index_name,
                "dimension": dimension,
                "index_fullness": stats.get("index_fullness"),
                "total_vector_count": stats.get("total_vector_count"),
                "namespaces": stats.get("namespaces", {}),
                "config": config_dump,
            }
        except Exception as e:
            logger.error("Failed to get index stats: %s", e)
            raise RAGError(f"Unable to retrieve index statistics: {e}") from e
