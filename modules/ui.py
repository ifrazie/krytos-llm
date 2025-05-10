import streamlit as st
import logging
from datetime import datetime
from typing import List, Dict, Any

def apply_custom_css():
    """Apply custom CSS styles for the chat interface"""
    st.markdown(
        """
        <style>
        /* Global styling for chat messages */
        .stChatMessage .stChatMessageContent {
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        /* User message styling */
        .stChatMessage.user .stChatMessageContent {
            background-color: #e6f7ff !important;
            border: 1px solid #91d5ff !important;
        }
        /* Assistant message styling */
        .stChatMessage.assistant .stChatMessageContent {
            background-color: #fffbe6 !important;
            border: 1px solid #ffe58f !important;
        }
        /* System message styling */
        .stChatMessage.system .stChatMessageContent {
            background-color: #f6ffed !important;
            border: 1px solid #b7eb8f !important;
        }
        /* Tool message styling */
        .stChatMessage.tool .stChatMessageContent {
            background-color: #f9f0ff !important;
            border: 1px solid #d3adf7 !important;
        }
        /* Error message styling */
        .error-message {
            color: #ff4d4f;
            font-weight: bold;
        }
        /* Session selector styling */
        .stSelectbox {
            margin-top: 10px;
            margin-bottom: 10px;
        }
        /* Success message styling */
        .success-message {
            color: #52c41a;
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_sidebar():
    """Render the sidebar with document upload, settings, and session management"""
    with st.sidebar:
        st.header("Document Upload")
        from modules.document_handler import handle_document_upload
        handle_document_upload()
        
        # Add function calling toggle
        st.header("Settings")
        st.session_state.function_calling_enabled = st.toggle(
            "Enable Function Calling",
            value=st.session_state.function_calling_enabled,
            help="Toggle to enable/disable function calling capabilities"
        )

        # Add session management UI
        st.header("Session Management")
        
        from modules.session_manager import create_new_session, switch_session, get_session_options
        
        if st.button("Start New Dossier"):
            create_new_session()
            st.success("New dossier started!")
        
        # Display sessions with a more descriptive label
        session_options = get_session_options()
        
        # Create the selectbox with friendly names but keep track of the session IDs
        session_ids = list(session_options.keys())
        session_names = list(session_options.values())
        current_index = session_ids.index(st.session_state.current_session_id)
        selected_index = st.selectbox(
            "Switch Dossier", 
            range(len(session_ids)), 
            format_func=lambda i: session_names[i],
            index=current_index
        )
        
        if session_ids[selected_index] != st.session_state.current_session_id:
            switch_session(session_ids[selected_index])
            st.rerun()  # Force a rerun to refresh the UI with the new session

def render_chat_history(messages: List[Dict[str, Any]]):
    """
    Render the chat history
    
    Args:
        messages: The chat messages to render
    """
    # Group consecutive assistant messages to prevent empty chat bubbles
    i = 0
    while i < len(messages):
        message = messages[i]
        role = message["role"]
        content = message["content"]
        
        # Skip tool messages entirely
        if role == "tool":
            i += 1
            continue
            
        # If this is an assistant message, check for any following assistant messages
        # that might be generated after tool calls
        if role == "assistant" and i + 2 < len(messages):
            # Check if there's a tool message followed by another assistant message
            if (messages[i+1]["role"] == "tool" and 
                messages[i+2]["role"] == "assistant"):
                # Skip this message as we'll display the later assistant message
                i += 1
                continue
        
        with st.chat_message(role):
            st.markdown(content)
        
        i += 1

def render_model_selector(default_model_index: int = 0):
    """
    Render the model selector
    
    Args:
        default_model_index: The index of the default model to select
        
    Returns:
        str: The selected model name
    """
    from modules.model_manager import get_ollama_models
    
    available_models = get_ollama_models()
    return st.sidebar.selectbox("Choose a model", available_models, index=default_model_index)

def render_chat_download_button(model: str, messages: List[Dict[str, Any]]):
    """
    Render a download button for the chat history
    
    Args:
        model: The model used for the conversation
        messages: The conversation message history
    """
    from modules.model_manager import export_chat_history
    
    chat_history = export_chat_history(model, messages)
    if chat_history:
        st.sidebar.download_button(
            label="Download Chat History",
            data=chat_history,
            file_name=f"chat_history_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )