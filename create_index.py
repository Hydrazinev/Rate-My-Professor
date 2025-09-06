# create_index.py (Pinecone v5 client)
from pinecone import Pinecone, ServerlessSpec
import os, time

INDEX_NAME = "professors-index"          # must match your code
DIMENSION  = 1536                         # <-- replace with your actual dim
METRIC     = "cosine"                     # usually cosine for embeddings
CLOUD      = "aws"
REGION     = "us-west-2"                  # or your preferred region

api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    raise RuntimeError("Set PINECONE_API_KEY first.")

pc = Pinecone(api_key=api_key)

# Create if it doesn't exist
if INDEX_NAME not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric=METRIC,
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
    print(f"Creating index '{INDEX_NAME}'...")
else:
    print(f"Index '{INDEX_NAME}' already exists.")

# Wait until ready
while True:
    desc = pc.describe_index(INDEX_NAME)
    if desc.status.get("ready"):
        print("Index is ready.")
        break
    print("Waiting for index to be ready...")
    time.sleep(2)
