import streamlit as st
import logging
import os
import time
import subprocess
import json
from datetime import datetime
import asyncio
import ollama
from ollama import ChatResponse
import yaml
from typing import List, Dict, Any, Tuple, Optional
from pymilvus import connections, Collection, utility
import uuid

# Import the functions and dictionary from tool_functions.py
from tool_functions import available_functions
from document_loader import DocumentLoader, PDFProcessingError, MilvusConnectionError
from async_tools import process_tool_calls, format_tool_info_for_display

logging.basicConfig(level=logging.INFO)

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

# Initialize session state
if 'model_messages' not in st.session_state:
    st.session_state.model_messages = {}
if 'document_collection' not in st.session_state:
    st.session_state.document_collection = None
if 'milvus_connected' not in st.session_state:
    st.session_state.milvus_connected = False
if 'milvus_lite_db_file' not in st.session_state:
    st.session_state.milvus_lite_db_file = "milvus_app.db"  # Default DB file name

# Add this after other session state initializations
if 'function_calling_enabled' not in st.session_state:
    st.session_state.function_calling_enabled = False

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
        'selected_model': "digiguru/krytos:latest"  # Set default model to digiguru/krytos:latest
    }

# Connect to Milvus
def connect_to_milvus(host="localhost", port="19530", use_lite=True, db_file="milvus_app.db"):
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

# Function to create a new session (dossier)
def create_new_session():
    session_id = str(uuid.uuid4())
    st.session_state.sessions[session_id] = {
        'name': f"Dossier {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        'created_at': datetime.now().isoformat(),
        'model_messages': {},
        'document_collection': None,
        'selected_model': "digiguru/krytos:latest"  # Set default model to digiguru/krytos:latest
    }
    st.session_state.current_session_id = session_id
    logging.info(f"Created new session: {session_id}")
    return session_id

# Function to switch to another session
def switch_session(session_id: str):
    if session_id in st.session_state.sessions:
        st.session_state.current_session_id = session_id
        logging.info(f"Switched to session: {session_id}")
        return True
    logging.error(f"Session not found: {session_id}")
    return False

# Get current session data
def get_current_session() -> Dict:
    return st.session_state.sessions.get(st.session_state.current_session_id, {})

# Add this section for document upload and processing
def handle_document_upload():
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

if 'messages' not in st.session_state:
    st.session_state.messages = []

def query_documents(query: str, collection_name: str):
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
        
        # Get vector embedding for the query (this assumes your DocumentLoader has this functionality)
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

def load_tool_configs() -> List[Dict[str, Any]]:
    try:
        with open('tools_config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        return [{
            "type": "function",
            "function": tool
        } for tool in config['tools']]
    except Exception as e:
        logging.error(f"Error loading tool configs: {str(e)}")
        display_error(e, ErrorCategory.TOOL)
        return []

async def verify_model_health(model_name: str) -> bool:
    """Verify if the model is properly loaded and functioning."""
    try:
        client = ollama.AsyncClient()
        await client.chat(
            model=model_name,
            messages=[{"role": "user", "content": "test"}],
            stream=False
        )
        return True
    except Exception as e:
        logging.error(f"Model health check failed for {model_name}: {str(e)}")
        display_error(e, ErrorCategory.MODEL)
        return False

def get_ollama_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if result.returncode != 0:
            logging.error("Failed to get Ollama models list")
            return ["llama3.1:latest"]  # Fallback to a common model
            
        lines = result.stdout.strip().split('\n')[1:]
        models = [line.split()[0] for line in lines if line]
        
        if not models:
            logging.warning("No models found, using default model")
            return ["llama3.1:latest"]  # Fallback to a common model
            
        return models
    except Exception as e:
        logging.error(f"Error getting Ollama models: {str(e)}")
        display_error(e, ErrorCategory.MODEL)
        return ["llama3.1:latest"]  # Fallback to a common model

async def stream_chat_with_tools(model, messages):
    tool = None
    final_response = None
    tool_info = None
    try:
        client = ollama.AsyncClient()
        tools = load_tool_configs() if st.session_state.function_calling_enabled else None

        response: ChatResponse = await client.chat(
            model,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            tools=tools
        )

        # If function calling is enabled and we have tool calls, process them
        if st.session_state.function_calling_enabled and response.message.tool_calls:
            response_content, tool_info = await process_tool_calls(client, model, messages, response)
            return response_content, tool_info
        
        # No tool calls, just return the response
        return response.message.content, None
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        display_error(e, ErrorCategory.TOOL if "tool" in str(e).lower() else ErrorCategory.MODEL)
        raise e

def serialize_message(message):
    """Convert a message object to a serializable format."""
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

def export_chat_history(model, messages):
    if not messages:
        return None

    chat_data = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "messages": [serialize_message(msg) for msg in messages]
    }

    return json.dumps(chat_data, indent=2)

def main():
    st.title("Krytos AI")

    # Apply custom CSS for styling chat messages - moved to top level to ensure it's always applied
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

    # Document upload section
    with st.sidebar:
        st.header("Document Upload")
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
        if st.button("Start New Dossier"):
            create_new_session()
            st.success("New dossier started!")
        
        # Display sessions with a more descriptive label
        session_options = {}
        for session_id in st.session_state.sessions:
            session_data = st.session_state.sessions[session_id]
            created_at = datetime.fromisoformat(session_data['created_at']).strftime('%Y-%m-%d %H:%M')
            message_count = sum(len(msgs) for msgs in session_data['model_messages'].values())
            session_options[session_id] = f"{session_data['name']} ({message_count} messages)"
        
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

    logging.info("App started")

    # Update the tools loading section to show status based on toggle
    tools = load_tool_configs() if st.session_state.function_calling_enabled else []
    if st.session_state.function_calling_enabled:
        if not tools:
            st.sidebar.warning("No tool configurations loaded")
        else:
            st.sidebar.success(f"Loaded {len(tools)} tools")

    available_models = get_ollama_models()
    
    current_session = get_current_session()
    
    # Get the default model index - use the session's saved model if available, otherwise use the first model
    default_model_index = 0
    if current_session['selected_model'] in available_models:
        default_model_index = available_models.index(current_session['selected_model'])
    
    # Select model - use the session's saved model if available
    model = st.sidebar.selectbox("Choose a model", available_models, index=default_model_index)
    
    # Save the selected model with the current session
    current_session['selected_model'] = model
    
    # Add model health check
    if not asyncio.run(verify_model_health(model)):
        st.error(f"Error: Model '{model}' is not responding properly. Try pulling the model again using 'ollama pull {model}'")
        st.stop()
    
    logging.info(f"Model selected and verified: {model}")

    current_session = get_current_session()
    if model in current_session['model_messages'] and current_session['model_messages'][model]:
        chat_history = export_chat_history(model, current_session['model_messages'][model])
        if chat_history:
            st.sidebar.download_button(
                label="Download Chat History",
                data=chat_history,
                file_name=f"chat_history_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    if model not in current_session['model_messages']:
        current_session['model_messages'][model] = []

    current_messages = current_session['model_messages'][model]

    # Display current dossier info
    st.info(f"Current Dossier: {current_session['name']}")

    # Render chat messages - improved to work consistently across session switches
    # Group consecutive assistant messages to prevent empty chat bubbles
    i = 0
    while i < len(current_messages):
        message = current_messages[i]
        role = message["role"]
        content = message["content"]
        
        # Skip tool messages entirely
        if role == "tool":
            i += 1
            continue
            
        # If this is an assistant message, check for any following assistant messages
        # that might be generated after tool calls
        if role == "assistant" and i + 2 < len(current_messages):
            # Check if there's a tool message followed by another assistant message
            if (current_messages[i+1]["role"] == "tool" and 
                current_messages[i+2]["role"] == "assistant"):
                # Skip this message as we'll display the later assistant message
                i += 1
                continue
        
        with st.chat_message(role):
            st.markdown(content)
        
        i += 1

    # Wrap processing logic with st.spinner to show a loading indicator
    if prompt := st.chat_input("Your question"):
        with st.chat_message("user"):
            st.markdown(prompt)
        current_messages.append({"role": "user", "content": prompt})
        logging.info(f"User input: {prompt}")

        if current_messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                logging.info("Generating response")

                with st.spinner("Processing your input..."):
                    try:
                        # Get relevant documents if we have a collection
                        if current_session['document_collection']:
                            try:
                                relevant_docs = query_documents(current_messages[-1]["content"],
                                                           current_session['document_collection'])
                                if relevant_docs and relevant_docs['documents'] and relevant_docs['documents'][0]:
                                    context = "\nContext from documents:\n" + "\n".join(relevant_docs['documents'][0])
                                    # Add context to the messages
                                    current_messages.append({
                                        "role": "system",
                                        "content": f"Here's relevant context for the query: {context}"
                                    })
                                    logging.info(f"Added context from Milvus: {len(relevant_docs['documents'][0])} documents")
                            except Exception as e:
                                logging.error(f"Error retrieving document context: {str(e)}")
                                display_error(e, ErrorCategory.DOCUMENT)
                                # Continue without context if there's an error
                                
                        # Process the chat with potential tool calls
                        response_message, tool_info = asyncio.run(stream_chat_with_tools(model, current_messages))
                        
                        # Calculate duration for display
                        duration = time.time() - start_time
                        
                        # Format the full response to display to the user
                        display_message = response_message
                        
                        # Add tool info if available
                        tool_display = format_tool_info_for_display(tool_info) if tool_info else ""
                        
                        # Add the complete response to the message history without tool details
                        current_messages.append({
                            "role": "assistant",
                            "content": response_message
                        })
                        
                        # Display the response and duration
                        st.markdown(display_message)
                        if tool_info:
                            with st.expander("Tool Usage Details", expanded=False):
                                for info in tool_info:
                                    st.markdown(info)
                        st.write(f"Duration: {duration:.2f} seconds")
                        logging.info(f"Response generated, Duration: {duration:.2f} s")

                    except Exception as e:
                        error_message = str(e)
                        logging.error(f"Error: {error_message}")
                        
                        # Add a simple error message to the chat history
                        current_messages.append({
                            "role": "assistant",
                            "content": "I encountered an error processing your request."
                        })
                        
                        # Display a user-friendly error to the user
                        if "model" in error_message.lower():
                            display_error(e, ErrorCategory.MODEL)
                        elif "connection" in error_message.lower() or "timeout" in error_message.lower():
                            display_error(e, ErrorCategory.CONNECTION)
                        elif "tool" in error_message.lower():
                            display_error(e, ErrorCategory.TOOL)
                        else:
                            display_error(e, ErrorCategory.UNKNOWN)

if __name__ == "__main__":
    main()