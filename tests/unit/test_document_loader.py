import pytest
import json
import numpy as np
from unittest.mock import patch, MagicMock, mock_open

from document_loader import DocumentLoader

class TestDocumentLoader:
    """Test the DocumentLoader class and its methods"""
    
    def test_init(self):
        """Test the initialization of DocumentLoader"""
        # Create instance in testing mode
        loader = DocumentLoader(testing_mode=True)
        
        # Verify instance attributes
        assert loader.text_splitter is not None
        assert loader.testing_mode is True
        assert loader.embeddings is None  # Should be None in testing mode
    
    def test_load_single_pdf(self):
        """Test loading a single PDF file"""
        # Create DocumentLoader instance in testing mode
        loader = DocumentLoader(testing_mode=True)
        
        # Call the method - in testing mode it returns mock data
        result = loader.load_single_pdf("test.pdf")
        
        # Assertions
        assert len(result) == 3
        assert result[0].page_content == "This is mock PDF content for testing."
        assert result[0].metadata["source"] == "test.pdf"
    
    def test_get_embeddings_testing_mode(self):
        """Test generating embeddings in testing mode"""
        # Create instance in testing mode
        loader = DocumentLoader(testing_mode=True)
        
        # Call the method - should return mock embeddings
        result = loader.get_embeddings(["text1", "text2"])
        
        # Assertions
        assert len(result) == 2
        assert all(len(emb) == 3 for emb in result)  # Mock embeddings are [0.1, 0.2, 0.3]
        assert result[0] == [0.1, 0.2, 0.3]
    
    def test_setup_milvus_testing_mode(self):
        """Test setting up a Milvus collection in testing mode"""
        # Create loader instance in testing mode
        loader = DocumentLoader(testing_mode=True)
        
        # Create mock documents
        class MockDocument:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata
        
        documents = [
            MockDocument("content1", {"source": "test1.pdf", "page": 1}),
            MockDocument("content2", {"source": "test1.pdf", "page": 2})
        ]
        
        # Call the method - in testing mode it just returns the collection name
        result = loader.setup_milvus(documents, "test_collection")
        
        # Assertions
        assert result == "test_collection"
        
    def test_query_documents_testing_mode(self):
        """Test querying documents in testing mode"""
        # Create loader instance in testing mode
        loader = DocumentLoader(testing_mode=True)
        
        # Call the method - in testing mode it returns mock results
        result = loader.query_documents("test query", "test_collection")
        
        # Assertions
        assert len(result) == 2
        assert result[0] == "Test document 1"
        assert result[1] == "Test document 2"
        
    # Optional: Add tests for non-testing mode with proper mocking
    @patch('pymilvus.connections.connect')
    @patch('langchain_community.embeddings.HuggingFaceEmbeddings')
    def test_get_embeddings_real_mode(self, mock_embeddings_class, mock_connect):
        """Test generating embeddings in real mode with mocking"""
        # Setup mock embeddings
        mock_embeddings_instance = MagicMock()
        mock_embeddings_instance.embed_documents.return_value = [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        mock_embeddings_class.return_value = mock_embeddings_instance
        
        # Create instance with mocked embeddings
        loader = DocumentLoader()  # Not in testing mode
        loader.embeddings = mock_embeddings_instance  # Replace with mock
        
        # Call the method
        result = loader.get_embeddings(["text1", "text2"])
        
        # Assertions
        mock_embeddings_instance.embed_documents.assert_called_once_with(["text1", "text2"])
        assert result == [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]