import os
import json
import logging
import numpy as np
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymilvus import (
    connections,
    utility,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType
)

class DocumentLoader:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        logging.info("DocumentLoader initialized")
        self.connect_to_milvus()

    def connect_to_milvus(self):
        connections.connect(host='localhost', port='19530')

    def load_single_pdf(self, file_path: str):
        """Load a single PDF and split it into chunks"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            chunks = self.text_splitter.split_documents(documents)
            logging.info(f"Loaded PDF with {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logging.error(f"Error loading PDF: {str(e)}")
            raise e
            
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            raise e

    def setup_milvus(self, documents: List, collection_name: str) -> str:
        """Set up a Milvus collection and insert document embeddings"""
        try:
            # Extract text from documents
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Generate embeddings
            embeddings = self.get_embeddings(texts)
            
            # Create collection schema
            dim = len(embeddings[0])  # Get embedding dimension
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
            ]
            
            schema = CollectionSchema(fields)
            
            # Create collection if it doesn't exist
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                
            collection = Collection(name=collection_name, schema=schema)
            
            # Create index for vector field
            index_params = {
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 8, "efConstruction": 64}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            
            # Insert data
            data = [
                texts,                              # content
                [json.dumps(meta) for meta in metadatas],  # metadata as JSON
                embeddings                          # embedding vectors
            ]
            
            collection.insert(data)
            collection.flush()
            logging.info(f"Created Milvus collection '{collection_name}' with {len(texts)} documents")
            
            return collection_name
            
        except Exception as e:
            logging.error(f"Error setting up Milvus: {str(e)}")
            raise e
            
    def setup_chroma(self, documents):
        """Legacy method for Chroma DB support - kept for compatibility"""
        logging.warning("setup_chroma called - Milvus should be used instead")
        raise NotImplementedError("Chroma DB support has been replaced by Milvus")
        
    def query_documents(self, query: str, collection_name: str = "documents", top_k: int = 3):
        collection = Collection(collection_name)
        collection.load()
        
        query_embedding = self.embeddings.embed_documents([query])[0]
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["content"]
        )
        
        return [hit.entity.get('content') for hit in results[0]]
