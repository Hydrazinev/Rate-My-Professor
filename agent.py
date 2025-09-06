"""
Professor Rating Agent with Pinecone RAG + RateMyProfessors fallback.
Behavior is intentionally unchanged; this refactor improves structure,
typing, naming, and documentation.
"""
from __future__ import annotations

from functools import cached_property
from typing import Generator, Sequence

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pinecone import Pinecone
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag import RAG

# 🔗 RMP fallback tools
# NOTE: matches earlier module: tools/ratemyprofessors.py
from tools.ratemyprofessor import run_tools as rmp_run


# =============================================================================
# Configuration models
# =============================================================================

class AgentConfig(BaseSettings):
    """Configuration settings for the ProfessorRaterAgent."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    pinecone_api_key: str = Field(..., description="Pinecone API key")
    openai_api_key: str = Field(..., description="OpenAI API key")
    pinecone_index_name: str = Field(
        default="professors-index",
        description="Pinecone index name",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI LLM model",
    )


class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    answer: str
    found_professors: bool = True


# =============================================================================
# Core agent
# =============================================================================

class ProfessorRaterAgent:
    """
    Retrieval-augmented agent that searches Pinecone first,
    then falls back to RateMyProfessors when no result is found.
    """

    NO_PROFESSOR_MARKER = "NO PROFESSOR"

    def __init__(self, config: AgentConfig | None = None) -> None:
        self._config = config or AgentConfig()
        if not self._config.pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY environment variable is required but not set"
            )
        # OPENAI_API_KEY is read by langchain_openai under the hood

    # ---------- Lazy resources ----------

    @cached_property
    def _pinecone_client(self) -> Pinecone:
        return Pinecone(api_key=self._config.pinecone_api_key)

    @cached_property
    def _embeddings(self) -> OpenAIEmbeddings:
        return OpenAIEmbeddings(model=self._config.embedding_model)

    @cached_property
    def _rag(self) -> RAG:
        return RAG(
            pinecone_client=self._pinecone_client,
            pinecone_index_name=self._config.pinecone_index_name,
            embedding=self._embeddings,
        )

    @cached_property
    def _llm(self) -> ChatOpenAI:
        # temperature=0 for more deterministic retrieval answers
        return ChatOpenAI(model=self._config.llm_model, temperature=0)

    # ---------- Prompts & chains ----------

    @cached_property
    def _contextualize_prompt(self) -> ChatPromptTemplate:
        system_prompt = (
            "Given a chat history and the latest user question which might reference "
            "context in the chat history, formulate a standalone question which can be "
            "understood without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    @cached_property
    def _qa_prompt(self) -> ChatPromptTemplate:
        system_prompt = (
            "You are the Professor Finder Bot, created by Vaidik. Your primary role is to "
            "help users find professors from an internal database. If you can't locate the "
            "information there, you can use specialized tools to search by professor names or "
            "their associated universities.\n\n"
            "Use the provided context to identify professors that meet the user's criteria. "
            f"If you don't find any relevant professors, respond with \"{self.NO_PROFESSOR_MARKER}\".\n\n"
            "Communication Guidelines:\n"
            "- Use a friendly and approachable tone\n"
            "- Avoid overly formal or technical language\n"
            "- Be clear and concise in your responses\n\n"
            f"Important: If no professors are found or not enough information is found, "
            f"always end your response with \"{self.NO_PROFESSOR_MARKER}\".\n\n"
            "Context: {context}"
        )
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

    @cached_property
    def _retrieval_chain(self):
        """Construct a history-aware retrieval chain (cached)."""
        history_aware_retriever = create_history_aware_retriever(
            self._llm, self._rag.get_retriever(), self._contextualize_prompt
        )
        question_answer_chain = create_stuff_documents_chain(
            self._llm, self._qa_prompt
        )
        return create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # ---------- Core API ----------

    def _pinecone_first_then_rmp(
        self,
        user_input: str,
        chat_history: Sequence[BaseMessage] | None,
    ) -> ChatResponse:
        """
        Try Pinecone RAG, and if not found, fall back to RateMyProfessors tools.
        """
        # Normalize to list for downstream calls (LangChain tools expect lists)
        history_list = list(chat_history or [])

        # 1) Pinecone RAG
        result = self._retrieval_chain.invoke(
            {"input": user_input, "chat_history": history_list}
        )
        answer: str = result["answer"]
        found_professors = self.NO_PROFESSOR_MARKER not in answer
        if found_professors:
            return ChatResponse(answer=answer, found_professors=True)

        # 2) Fallback: RateMyProfessors
        try:
            rmp_answer = rmp_run(user_input, history_list)
            if isinstance(rmp_answer, str) and rmp_answer.strip():
                # Optional: warm up and upsert fetched text back into Pinecone for next time
                try:
                    # Keep non-fatal; preserves existing behavior
                    _ = self._rag.lookup("warmup")
                    if hasattr(self._rag, "_vector_store") and hasattr(
                        self._rag._vector_store, "add_texts"
                    ):
                        self._rag._vector_store.add_texts(
                            texts=[rmp_answer],
                            metadatas=[{"source": "ratemyprofessors"}],
                            ids=[f"rmp-{abs(hash(user_input))}"],
                        )
                except Exception:
                    # Non-fatal if warmup/upsert fails
                    pass
                return ChatResponse(answer=rmp_answer, found_professors=True)
        except Exception:
            # Fallback failed; return original "not found" answer
            pass

        return ChatResponse(answer=answer, found_professors=False)

    def invoke(
        self,
        user_input: str,
        chat_history: Sequence[BaseMessage] | None = None,
    ) -> ChatResponse:
        """
        Process a user query and return a completed response (with RMP fallback if needed).
        """
        return self._pinecone_first_then_rmp(user_input, chat_history)

    def stream(
        self,
        user_input: str,
        chat_history: Sequence[BaseMessage] | None = None,
    ) -> Generator[str, None, None]:
        """
        Stream the response. For reliability of the RMP fallback, we compute the full
        answer first (via invoke) and then stream it as a single chunk.
        """
        resp = self._pinecone_first_then_rmp(user_input, chat_history)
        yield resp.answer

    # ---------- Optional CLI ----------

    def interactive_cli(self) -> None:
        """
        Minimal interactive loop for local testing.
        Note: This intentionally keeps the same message types as your original code
        (HumanMessage and SystemMessage) to avoid altering downstream behavior.
        """
        print("🎓 Professor Finder Bot - Type 'exit' to quit.")
        print("-" * 50)
        chat_history: list[BaseMessage] = []
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                if user_input.lower() in {"exit", "quit", "bye"}:
                    print("👋 Goodbye!")
                    break
                if not user_input:
                    continue
                resp = self.invoke(user_input, chat_history)
                print(f"🤖 Bot: {resp.answer}")
                chat_history.extend(
                    [HumanMessage(content=user_input), SystemMessage(content=resp.answer)]
                )
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")


# =============================================================================
# Factory
# =============================================================================

def create_agent(config_path: str | None = None) -> ProfessorRaterAgent:
    """
    Factory function to create a configured ProfessorRaterAgent.
    """
    config = AgentConfig(_env_file=config_path) if config_path else AgentConfig()
    return ProfessorRaterAgent(config)


# =============================================================================
# CLI entrypoint
# =============================================================================

if __name__ == "__main__":
    # Simple CLI test
    agent = create_agent()
    agent.interactive_cli()
