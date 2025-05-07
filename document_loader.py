import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from pymilvus import (
    utility,
    MilvusClient,
    DataType,
    FieldSchema
)

class DocumentLoader:
    def __init__(self, testing_mode: bool = False, milvus_host: str = 'localhost', milvus_port: str = '19530', use_lite: bool = True, milvus_db_file: str = "milvus_app.db"):
        """Initialize the DocumentLoader
        
        Args:
            testing_mode: If True, won't try to load real models or make real connections
            milvus_host: Host for Milvus connection (if not using Lite)
            milvus_port: Port for Milvus connection (if not using Lite)
            use_lite: If True, use Milvus Lite
            milvus_db_file: DB file for Milvus Lite
        """
        self.testing_mode = testing_mode
        self.use_lite = use_lite
        self.milvus_db_file = milvus_db_file
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.client = None

        if not testing_mode:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
        else:
            self.embeddings = None
            logging.info("Initialized DocumentLoader in testing mode - no real connections made")
            
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        logging.info("DocumentLoader initialized")

    def connect(self) -> bool:
        """Establish connection to Milvus using MilvusClient."""
        if self.client:
            logging.info("MilvusClient already initialized.")
            return True

        if self.testing_mode:
            logging.info("In testing mode, skipping actual Milvus connection.")
            return True 

        try:
            uri_to_use = ""
            if self.use_lite:
                uri_to_use = self.milvus_db_file
                if not os.path.isabs(uri_to_use) and not uri_to_use.startswith(("./", ".\\")) and "/" not in uri_to_use and "\\" not in uri_to_use:
                    uri_to_use = "./" + uri_to_use
                logging.info(f"Attempting to connect to Milvus Lite with URI: {uri_to_use}")
            else:
                uri_to_use = f"http://{self.milvus_host}:{self.milvus_port}"
                logging.info(f"Attempting to connect to Milvus Standalone at URI: {uri_to_use}")
            
            self.client = MilvusClient(uri=uri_to_use)
            self.client.list_collections() 
            logging.info(f"Successfully connected to Milvus with MilvusClient using URI: {uri_to_use}")
            return True
        except Exception as e:
            logging.error(f"Failed to connect to Milvus with MilvusClient using URI {uri_to_use}: {str(e)}")
            self.client = None
            return False

    def load_single_pdf(self, file_path: str):
        """Load a single PDF and split it into chunks"""
        try:
            if self.testing_mode:
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
                return [[0.1, 0.2, 0.3] for _ in texts]
            return self.embeddings.embed_documents(texts)
        except Exception as e:
            logging.error(f"Error generating embeddings: {str(e)}")
            raise e

    def setup_milvus(self, documents: List, collection_name: str) -> str:
        """Set up a Milvus collection and insert document embeddings using MilvusClient."""
        if self.testing_mode:
            logging.info(f"Testing mode: Skipping actual Milvus setup for collection {collection_name}")
            return collection_name
        
        if not self.client and not self.connect():
            raise ConnectionError("Failed to connect to Milvus. Cannot setup collection.")

        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            embeddings = self.get_embeddings(texts)
            
            if not embeddings:
                logging.error("No embeddings generated. Cannot setup Milvus collection.")
                raise ValueError("Embeddings are empty.")

            dim = len(embeddings[0])

            if self.client.has_collection(collection_name):
                logging.info(f"Collection {collection_name} already exists. Dropping it.")
                self.client.drop_collection(collection_name)
            
            # Define schema for other fields
            other_fields = [
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]

            self.client.create_collection(
                collection_name=collection_name,
                dimension=dim,
                primary_field_name="id",        # Explicitly name primary key
                auto_id=True,                   # Enable auto ID generation
                vector_field_name="embedding",  # Name of the vector field
                other_fields_schema=other_fields, # Schema for other fields
                metric_type="COSINE",
                consistency_level="Strong",
            )
            logging.info(f"Collection {collection_name} created with dimension {dim}.")

            # Convert to row-oriented format instead of column-oriented
            insert_data_rows = []
            for i in range(len(texts)):
                row = {
                    "content": texts[i],
                    "metadata": json.dumps(metadatas[i]),  # Ensure metadata is JSON serialized
                    "embedding": embeddings[i]
                }
                insert_data_rows.append(row)
            
            self.client.insert(collection_name=collection_name, data=insert_data_rows)
            self.client.flush(collection_name=collection_name) # Ensure data is flushed
            logging.info(f"Successfully set up Milvus collection '{collection_name}' with {len(texts)} documents.")
            return collection_name
            
        except Exception as e:
            logging.error(f"Error setting up Milvus collection {collection_name}: {str(e)}")
            raise e
            
    def setup_chroma(self, documents):
        """Legacy method for Chroma DB support - kept for compatibility"""
        logging.warning("setup_chroma called - Milvus should be used instead")
        raise NotImplementedError("Chroma DB support has been replaced by Milvus")
        
    def query_documents(self, query: str, collection_name: str = "documents", top_k: int = 3) -> List[str]:
        """Query documents from the vector database using MilvusClient."""
        if self.testing_mode:
            logging.info(f"Testing mode: Returning mock documents for query '{query}' from {collection_name}")
            return ["Test document 1", "Test document 2"]
            
        if not self.client and not self.connect():
            logging.error("Failed to connect to Milvus. Cannot query documents.")
            return []

        try:
            if not self.client.has_collection(collection_name):
                logging.warning(f"Collection {collection_name} does not exist. Cannot query.")
                return []

            query_embedding = self.embeddings.embed_documents([query])[0]
            
            search_params = {
                "metric_type": "COSINE",
                "params": {}, 
            } 

            results = self.client.search(
                collection_name=collection_name,
                data=[query_embedding],
                limit=top_k,
                search_params=search_params,
                output_fields=["content", "metadata"] # Corrected output fields
            )
            
            documents_content = []
            if results and results[0]:
                for hit in results[0]:
                    doc_text = hit.get('entity', {}).get('content', '') # Corrected field name
                    documents_content.append(doc_text)
            
            logging.info(f"Query '{query}' on {collection_name} returned {len(documents_content)} documents.")
            return documents_content
        except Exception as e:
            logging.error(f"Error querying Milvus collection {collection_name}: {str(e)}")
            return []
