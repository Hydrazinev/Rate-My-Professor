# check_pinecone.py
import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
print("INDEXES:", [i.name for i in pc.list_indexes()])  # should include 'professors-index'

name = "professors-index"
if any(i.name == name for i in pc.list_indexes()):
    print("DESCRIBE:", pc.describe_index(name))
else:
    print(f"'{name}' not found.")
