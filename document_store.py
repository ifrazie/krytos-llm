from pymilvus import Collection, connections, utility
from typing import List, Dict, Any
import numpy as np
import logging

class MilvusDocumentStore:
    def __init__(self, collection_name: str = "documents"):
        self.collection_name = collection_name
        self.is_connected = self._verify_connection()
        if self.is_connected:
            self.setup_collection()

    def _verify_connection(self) -> bool:
        """Verify Milvus connection is available"""
        try:
            connections.connect("default", host="localhost", port="19530", timeout=5)
            return True
        except Exception as e:
            logging.warning(f"Failed to connect to Milvus: {str(e)}")
            return False

    def setup_collection(self):
        if not self.is_connected:
            raise ConnectionError("Milvus connection not available")
        if utility.exists_collection(self.collection_name):
            self.collection = Collection(self.collection_name)
        else:
            # Define collection schema
            from pymilvus import CollectionSchema, FieldSchema, DataType
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=1536)  # For OpenAI embeddings
            ]
            schema = CollectionSchema(fields=fields, description="Document store")
            self.collection = Collection(name=self.collection_name, schema=schema)
            # Create index for vector field
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="embeddings", index_params=index_params)

    def add_documents(self, texts: List[str], embeddings: List[List[float]]):
        if not self.is_connected:
            raise ConnectionError("Milvus connection not available")
        data = [
            [i for i in range(len(texts))],  # id
            texts,  # content
            embeddings  # embeddings
        ]
        self.collection.insert(data)
        self.collection.flush()

    def query(self, query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        if not self.is_connected:
            raise ConnectionError("Milvus connection not available")
        self.collection.load()
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = self.collection.search(
            data=[query_vector],
            anns_field="embeddings",
            param=search_params,
            limit=top_k,
            output_fields=["content"]
        )
        
        return [{"content": hit.entity.get('content'), "score": hit.distance} 
                for hit in results[0]]
