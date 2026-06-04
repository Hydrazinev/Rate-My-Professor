# create_index.py — one-time Pinecone index creation (Pinecone v5 client)
# Reads all config from environment variables so it matches the running app.
from pinecone import Pinecone, ServerlessSpec
import os
import time

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "professors-index")
DIMENSION  = 1536          # text-embedding-3-small; change if you swap models
METRIC     = "cosine"
CLOUD      = "aws"
REGION     = os.environ.get("PINECONE_REGION", "us-east-1")   # must match app config

api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    raise RuntimeError("Set PINECONE_API_KEY first.")

pc = Pinecone(api_key=api_key)

if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    print(f"Creating index '{INDEX_NAME}' (dim={DIMENSION}, region={REGION})...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
else:
    print(f"Index '{INDEX_NAME}' already exists.")

# Wait until ready
while True:
    desc = pc.describe_index(INDEX_NAME)
    status = getattr(desc, "status", None) or (desc.get("status") if isinstance(desc, dict) else {}) or {}
    if status.get("ready"):
        print("Index is ready.")
        break
    print("Waiting for index to be ready...")
    time.sleep(2)

