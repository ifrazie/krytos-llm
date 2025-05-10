import os
import logging
import streamlit as st
from datetime import datetime
from typing import Dict, Any, List
from pymilvus import connections, Collection, utility

from document_loader import DocumentLoader, PDFProcessingError, MilvusConnectionError
from modules.error_handler import display_error, ErrorCategory

def connect_to_milvus(host="localhost", port="19530", use_lite=True, db_file="milvus_app.db") -> bool:
    """
    Connect to Milvus database (either Standalone or Lite version)
    
    Args:
        host: Milvus host
        port: Milvus port
        use_lite: Whether to use Milvus Lite
        db_file: Database file for Milvus Lite
        
    Returns:
        bool: True if connected successfully, False otherwise
    """
    try:
        if use_lite:
            uri_to_use = db_file
            # If db_file is a simple name (no path separators) and not absolute, prepend "./"
            if not os.path.isabs(db_file) and not db_file.startswith(("./", ".\\")) and "/" not in db_file and "\\" not in db_file:
                uri_to_use = "./" + db_file
            
            connections.connect(alias="default", uri=uri_to_use)
            st.session_state.milvus_lite_db_file = db_file  # Store the original name for UI consistency
            logging.info(f"Successfully connected to Milvus Lite using file: {uri_to_use}")
        else:
            connections.connect("default", host=host, port=port)
            logging.info(f"Successfully connected to Milvus at {host}:{port}")
        st.session_state.milvus_connected = True
        return True
    except Exception as e:
        logging.error(f"Failed to connect to Milvus: {str(e)}")
        st.session_state.milvus_connected = False
        display_error(e, ErrorCategory.CONNECTION)
        return False

def handle_document_upload():
    """Display and manage document upload UI and processing"""
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    # Add Milvus connection settings
    with st.expander("Milvus Connection Settings"):
        use_milvus_lite = st.checkbox("Use Milvus Lite", value=True)
        milvus_db_file = st.text_input("Milvus Lite DB File", value=st.session_state.get("milvus_lite_db_file", "milvus_app.db"))
        
        if not use_milvus_lite:
            milvus_host = st.text_input("Milvus Host", "localhost")
            milvus_port = st.text_input("Milvus Port", "19530")
        else:
            milvus_host = None
            milvus_port = None

        if st.button("Connect to Milvus"):
            if use_milvus_lite:
                if connect_to_milvus(use_lite=True, db_file=milvus_db_file):
                    st.success(f"Connected to Milvus Lite using file: {milvus_db_file}!")
                else:
                    st.error("Failed to connect to Milvus Lite")
            else:
                if connect_to_milvus(host=milvus_host, port=milvus_port, use_lite=False):
                    st.success("Connected to Milvus server!")
                else:
                    st.error("Failed to connect to Milvus server")
    
    if uploaded_file is not None and st.session_state.milvus_connected:
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            # Initialize document loader
            loader = DocumentLoader()
            # Load and process the document
            documents = loader.load_single_pdf("temp.pdf")
            # Store documents in Milvus
            collection_name = "document_" + datetime.now().strftime("%Y%m%d%H%M%S")
            collection = loader.setup_milvus(documents, collection_name)
            
            from modules.session_manager import get_current_session
            current_session = get_current_session()
            current_session['document_collection'] = collection_name
            
            st.success(f"Document processed and stored in Milvus collection: {collection_name}")
        except Exception as e:
            display_error(e, ErrorCategory.DOCUMENT)
        finally:
            # Clean up temporary file
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")
    elif uploaded_file is not None and not st.session_state.milvus_connected:
        st.warning("Please connect to Milvus server first before uploading documents")

def query_documents(query: str, collection_name: str) -> Dict[str, List[str]]:
    """
    Query documents from the vector store
    
    Args:
        query: The search query
        collection_name: The collection to search in
        
    Returns:
        Dict: Dictionary with documents list
    """
    try:
        if not utility.has_collection(collection_name):
            logging.error(f"Collection {collection_name} does not exist")
            return {"documents": []}
            
        collection = Collection(collection_name)
        collection.load()
        
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }
        
        # Get vector embedding for the query
        loader = DocumentLoader()
        query_vector = loader.get_embeddings([query])[0]
        
        results = collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=3,
            output_fields=["content"]
        )
        
        documents = []
        if results and len(results) > 0:
            for hit in results[0]:
                documents.append(hit.entity.get('content'))
        
        return {"documents": [documents] if documents else []}
    except Exception as e:
        logging.error(f"Error querying Milvus: {str(e)}")
        display_error(e, ErrorCategory.MILVUS)
        return {"documents": []}