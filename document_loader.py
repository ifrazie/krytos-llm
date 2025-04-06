import os
from typing import List
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType
)
import numpy as np
from sentence_transformers import SentenceTransformer

class DocumentLoader:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.connect_to_milvus()
        
    def connect_to_milvus(self):
        connections.connect(host='localhost', port='19530')
        
    def create_collection(self, collection_name: str = "documents"):
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            
        dim = 384  # dimensionality of the sentence-transformer model
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim)
        ]
        schema = CollectionSchema(fields=fields, description="Document storage")
        collection = Collection(name=collection_name, schema=schema)
        
        # Create an IVF_FLAT index for the embeddings field
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 1024}
        }
        collection.create_index(field_name="embeddings", index_params=index_params)
        return collection
        
    def load_single_pdf(self, file_path: str) -> List[str]:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return text_splitter.split_documents(documents)
        
    def setup_milvus(self, documents, collection_name: str = "documents"):
        collection = self.create_collection(collection_name)
        
        texts = [doc.page_content for doc in documents]
        embeddings = self.encoder.encode(texts)
        
        entities = [
            {"text": texts[i], "embeddings": embeddings[i].tolist()}
            for i in range(len(texts))
        ]
        
        collection.insert(entities)
        collection.flush()
        return collection
        
    def query_documents(self, query: str, collection_name: str = "documents", top_k: int = 3):
        collection = Collection(collection_name)
        collection.load()
        
        query_embedding = self.encoder.encode([query])[0]
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embeddings",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )
        
        return [hit.entity.get('text') for hit in results[0]]
