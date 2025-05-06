import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
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
    def __init__(self, testing_mode: bool = False, milvus_host: str = 'localhost', milvus_port: str = '19530'):
        """Initialize the DocumentLoader
        
        Args:
            testing_mode: If True, won't try to load real models or make real connections
            milvus_host: Host for Milvus connection
            milvus_port: Port for Milvus connection
        """
        self.testing_mode = testing_mode
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        
        if not testing_mode:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            self.connect_to_milvus()
        else:
            # Create a stub for testing that won't try to load a real model
            self.embeddings = None
            logging.info("Initialized in testing mode - no real connections made")
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        logging.info("DocumentLoader initialized")

    def connect_to_milvus(self):
        """Connect to Milvus vector database"""
        connections.connect(host=self.milvus_host, port=self.milvus_port)

    def load_single_pdf(self, file_path: str):
        """Load a single PDF and split it into chunks"""
        try:
            if self.testing_mode:
                # Return mock data for testing
                mock_content = "This is mock PDF content for testing."
                mock_metadata = {"source": file_path, "page": 1}
                mock_document = type('MockDocument', (), {'page_content': mock_content, 'metadata': mock_metadata})
                mock_chunks = [mock_document for _ in range(3)]
                return mock_chunks
                
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
            if self.testing_mode:
                # Return mock embeddings for testing
                return [[0.1, 0.2, 0.3] for _ in texts]
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            raise e

    def setup_milvus(self, documents: List, collection_name: str) -> str:
        """Set up a Milvus collection and insert document embeddings"""
        try:
            if self.testing_mode:
                # Skip actual Milvus operations in testing mode
                return collection_name
                
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
        """Query documents from the vector database
        
        Args:
            query: The query text
            collection_name: The name of the collection to search
            top_k: Number of results to return
            
        Returns:
            List of document contents
        """
        if self.testing_mode:
            # Return mock results for testing
            return ["Test document 1", "Test document 2"]
            
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
