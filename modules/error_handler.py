import streamlit as st
import logging
from typing import Tuple, Optional

# Define error categories for better user feedback
class ErrorCategory:
    CONNECTION = "Connection Error"
    MODEL = "Model Error"
    DOCUMENT = "Document Processing Error"
    TOOL = "Tool Execution Error"
    MILVUS = "Database Error"
    UNKNOWN = "Unknown Error"

def format_user_friendly_error(error: Exception, category: str = ErrorCategory.UNKNOWN) -> Tuple[str, str, Optional[str]]:
    """
    Format an exception into a user-friendly error message with potential solution.
    
    Args:
        error: The exception that occurred
        category: Error category for better classification
        
    Returns:
        Tuple containing (short_message, detailed_message, suggested_solution)
    """
    error_str = str(error)
    
    # Default values
    short_message = f"{category}: An error occurred"
    detailed_message = error_str
    suggested_solution = None
    
    # Customize error messages based on category and specific error types
    if category == ErrorCategory.CONNECTION:
        if "Connection refused" in error_str or "Failed to connect" in error_str:
            short_message = "Connection Failed"
            detailed_message = "Unable to connect to the required service."
            suggested_solution = "Please check that Ollama is running and accessible."
        elif "timeout" in error_str.lower():
            short_message = "Connection Timeout"
            detailed_message = "The connection timed out while waiting for a response."
            suggested_solution = "Check your network connection or try again later."
            
    elif category == ErrorCategory.MODEL:
        if "not found" in error_str.lower():
            short_message = "Model Not Available"
            detailed_message = "The selected AI model could not be found."
            suggested_solution = f"Try pulling the model using 'ollama pull [model_name]' in your terminal."
        elif "memory" in error_str.lower() or "resources" in error_str.lower():
            short_message = "Insufficient Resources"
            detailed_message = "The model requires more resources than are currently available."
            suggested_solution = "Try using a smaller model or closing other applications."
            
    elif category == ErrorCategory.DOCUMENT:
        if "pdf" in error_str.lower():
            short_message = "PDF Processing Failed"
            detailed_message = "Unable to process the PDF document."
            suggested_solution = "Ensure the PDF is not corrupted, password-protected, or in an unsupported format."
        elif "embedding" in error_str.lower():
            short_message = "Embedding Generation Failed"
            detailed_message = "Failed to generate embeddings for the document."
            suggested_solution = "Check that your embedding model is properly configured."
            
    elif category == ErrorCategory.MILVUS:
        if "collection" in error_str.lower():
            short_message = "Collection Error"
            detailed_message = "Problem accessing or creating the document collection."
            suggested_solution = "Try reconnecting to Milvus or creating a new collection."
        elif "search" in error_str.lower():
            short_message = "Search Failed"
            detailed_message = "The search operation in the vector database failed."
            suggested_solution = "Check your query or try reconnecting to the database."
            
    elif category == ErrorCategory.TOOL:
        if "permission" in error_str.lower():
            short_message = "Permission Denied"
            detailed_message = "The tool function lacks the necessary permissions."
            suggested_solution = "Check that the application has the required system permissions."
        elif "timeout" in error_str.lower():
            short_message = "Tool Execution Timeout"
            detailed_message = "The tool function took too long to execute."
            suggested_solution = "Try again with simplified parameters or check your network connection."
    
    return short_message, detailed_message, suggested_solution

def display_error(error: Exception, category: str = ErrorCategory.UNKNOWN):
    """
    Display a user-friendly error message in the Streamlit UI.
    
    Args:
        error: The exception that occurred
        category: Error category for better classification
    """
    short_message, detailed_message, suggested_solution = format_user_friendly_error(error, category)
    
    # Log the error for debugging
    logging.error(f"{category}: {detailed_message}")
    
    # Display error in UI with different levels of detail
    error_container = st.error(short_message)
    with error_container:
        with st.expander("Details", expanded=False):
            st.write(detailed_message)
            if suggested_solution:
                st.write("**Suggested Solution:**")
                st.write(suggested_solution)