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
from typing import List, Dict, Any
from pymilvus import connections, Collection, utility
import uuid

# Import the functions and dictionary from tool_functions.py
from tool_functions import available_functions
from document_loader import DocumentLoader

logging.basicConfig(level=logging.INFO)

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
            st.error(f"Error processing document: {str(e)}")
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
        return ["llama3.1:latest"]  # Fallback to a common model

async def stream_chat_with_tools(model, messages):
    tool = None
    final_response = None
    try:
        client = ollama.AsyncClient()
        tools = load_tool_configs() if st.session_state.function_calling_enabled else None

        response: ChatResponse = await client.chat(
            model,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            tools=tools
        )

        tool_output = None
        tool_info = []

        if st.session_state.function_calling_enabled and response.message.tool_calls:
            for tool in response.message.tool_calls:
                if function_to_call := available_functions.get(tool.function.name):
                    # Create tool call info string
                    tool_call_info = (
                        f"\n🔧 Tool Called: {tool.function.name}\n"
                        f"📝 Parameters: {json.dumps(tool.function.arguments, indent=2)}\n"
                    )

                    logging.info(f'Calling function: {tool.function.name}')
                    tool_output = function_to_call(**tool.function.arguments)

                    # Add result to tool info
                    tool_call_info += f"📊 Result: {tool_output}\n"
                    tool_info.append(tool_call_info)

                    logging.info(f'Function output: {tool_output}')

        if tool_output is not None:
            messages.append({"role": response.message.role, "content": response.message.content})
            messages.append({'role': 'tool', 'content': str(tool_output), 'name': tool.function.name})
            
            # Get final response from the model after seeing the tool output
            final_response = await client.chat(
                model, 
                messages=[{"role": m["role"], "content": m["content"]} for m in messages]
            )
            
            # Store just the model's response without tool details in the message history
            return final_response.message.content
        
        # Return only the model's response without any additional tool info
        return response.message.content
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
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
                        if current_session['document_collection']:
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
                        response_message = asyncio.run(stream_chat_with_tools(model, current_messages))
                        duration = time.time() - start_time
                        response_message_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
                        current_messages.append({
                            "role": "assistant",
                            "content": response_message_with_duration
                        })
                        st.markdown(response_message)
                        st.write(f"Duration: {duration:.2f} seconds")
                        logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")

                    except Exception as e:
                        current_messages.append({
                            "role": "assistant",
                            "content": str(e)
                        })
                        st.error("An error occurred while generating the response.")
                        logging.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()