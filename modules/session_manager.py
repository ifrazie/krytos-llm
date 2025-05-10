import streamlit as st
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional

def initialize_session_state():
    """Initialize all required session state variables"""
    if 'model_messages' not in st.session_state:
        st.session_state.model_messages = {}
    if 'document_collection' not in st.session_state:
        st.session_state.document_collection = None
    if 'milvus_connected' not in st.session_state:
        st.session_state.milvus_connected = False
    if 'milvus_lite_db_file' not in st.session_state:
        st.session_state.milvus_lite_db_file = "milvus_app.db"  # Default DB file name
    if 'function_calling_enabled' not in st.session_state:
        st.session_state.function_calling_enabled = False
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Add session management
    if 'sessions' not in st.session_state:
        st.session_state.sessions = {}
    if 'current_session_id' not in st.session_state:
        # Create initial session
        session_id = str(uuid.uuid4())
        st.session_state.current_session_id = session_id
        st.session_state.sessions[session_id] = {
            'name': f"Dossier {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            'created_at': datetime.now().isoformat(),
            'model_messages': {},
            'document_collection': None,
            'selected_model': "digiguru/krytos:latest"  # Set default model
        }

def create_new_session() -> str:
    """
    Create a new session (dossier)
    
    Returns:
        str: New session ID
    """
    session_id = str(uuid.uuid4())
    st.session_state.sessions[session_id] = {
        'name': f"Dossier {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        'created_at': datetime.now().isoformat(),
        'model_messages': {},
        'document_collection': None,
        'selected_model': "digiguru/krytos:latest"  # Set default model
    }
    st.session_state.current_session_id = session_id
    logging.info(f"Created new session: {session_id}")
    return session_id

def switch_session(session_id: str) -> bool:
    """
    Switch to another session
    
    Args:
        session_id: ID of the session to switch to
        
    Returns:
        bool: True if switch was successful, False otherwise
    """
    if session_id in st.session_state.sessions:
        st.session_state.current_session_id = session_id
        logging.info(f"Switched to session: {session_id}")
        return True
    logging.error(f"Session not found: {session_id}")
    return False

def get_current_session() -> Dict[str, Any]:
    """
    Get current session data
    
    Returns:
        Dict: The current session data
    """
    return st.session_state.sessions.get(st.session_state.current_session_id, {})

def get_session_options() -> Dict[str, str]:
    """
    Get all sessions with descriptive labels
    
    Returns:
        Dict: Dictionary with session IDs as keys and descriptive labels as values
    """
    session_options = {}
    for session_id in st.session_state.sessions:
        session_data = st.session_state.sessions[session_id]
        created_at = datetime.fromisoformat(session_data['created_at']).strftime('%Y-%m-%d %H:%M')
        message_count = sum(len(msgs) for msgs in session_data['model_messages'].values())
        session_options[session_id] = f"{session_data['name']} ({message_count} messages)"
    
    return session_options

def serialize_message(message: Any) -> Dict[str, Any]:
    """
    Convert a message object to a serializable format.
    
    Args:
        message: Message object to serialize
        
    Returns:
        Dict: Serialized message
    """
    if isinstance(message, dict):
        return {
            "role": message.get("role", ""),
            "content": message.get("content", ""),
            "name": message.get("name", "") if "name" in message else None
        }
    return {
        "role": getattr(message, "role", ""),
        "content": getattr(message, "content", ""),
        "name": getattr(message, "name", None) if hasattr(message, "name") else None
    }