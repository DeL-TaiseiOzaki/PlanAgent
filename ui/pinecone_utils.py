import os
from dotenv import load_dotenv
import config
import json
from typing import List, Dict, Any

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "gcp-starter")

class SimpleDictStore:
    def __init__(self):
        self.data = {}

    def upsert(self, vectors: List[tuple], namespace: str = "default"):
        for id, vector, metadata in vectors:
            self.data[id] = {"vector": vector, "metadata": metadata}

    def query(self, vector: List[float], top_k: int = 5, namespace: str = "default"):
        # 簡単な類似度計算（実際の実装ではより高度なアルゴリズムを使用すべきです）
        results = sorted(
            self.data.items(),
            key=lambda x: sum((a - b) ** 2 for a, b in zip(vector, x[1]["vector"])),
            reverse=True
        )[:top_k]
        return [{"id": id, "score": 1.0, "metadata": data["metadata"]} for id, data in results]

def init_vector_store():
    if config.USE_PINECONE:
        from pinecone import Pinecone, ServerlessSpec
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )
        return pc.Index(config.PINECONE_INDEX_NAME)
    else:
        return SimpleDictStore()

def upsert_to_store(store: Any, vectors: List[tuple], namespace: str = "default"):
    if config.USE_PINECONE:
        store.upsert(vectors=vectors, namespace=namespace)
    else:
        store.upsert(vectors, namespace)

def query_store(store: Any, query_vector: List[float], top_k: int = 5, namespace: str = "default"):
    if config.USE_PINECONE:
        return store.query(vector=query_vector, top_k=top_k, namespace=namespace)
    else:
        return store.query(query_vector, top_k, namespace)