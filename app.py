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

# Import the functions and dictionary from tool_functions.py
from tool_functions import available_functions
from document_loader import DocumentLoader

logging.basicConfig(level=logging.INFO)

# Initialize session state
if 'model_messages' not in st.session_state:
    st.session_state.model_messages = {}
if 'document_collection' not in st.session_state:
    st.session_state.document_collection = None

# Add this after other session state initializations
if 'function_calling_enabled' not in st.session_state:
    st.session_state.function_calling_enabled = False

# Add this section for document upload and processing
def handle_document_upload():
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            # Initialize document loader
            loader = DocumentLoader()
            # Load and process the document
            documents = loader.load_single_pdf("temp.pdf")
            # Setup Chroma and store documents
            collection = loader.setup_chroma(documents)
            st.session_state.document_collection = collection
            st.success("Document processed and stored successfully!")
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists("temp.pdf"):
                os.remove("temp.pdf")

if 'messages' not in st.session_state:
    st.session_state.messages = []

def query_documents(query: str, collection):
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    return results

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

                    # Format tool output based on type
                    if isinstance(tool_output, (list, dict)):
                        formatted_output = json.dumps(tool_output, indent=2)
                    else:
                        formatted_output = str(tool_output)

                    # Add result to tool info
                    tool_call_info += f"📊 Result: {formatted_output}\n"
                    tool_info.append(tool_call_info)

                    logging.info(f'Function output: {tool_output}')

        if tool_output is not None:
            messages.append(response.message)
            # Convert tool output to string format for the message
            tool_output_str = json.dumps(tool_output) if isinstance(tool_output, (list, dict)) else str(tool_output)
            messages.append({'role': 'tool', 'content': tool_output_str, 'name': tool.function.name})
            final_response = await client.chat(model, messages=[{"role": m["role"], "content": m["content"]} for m in messages])
            complete_response = final_response.message.content
            if tool_info:
                complete_response += "\n\n💡 Tool Usage Details:" + "".join(tool_info)
            return complete_response

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
    st.title("PenTestLLM")

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

    logging.info("App started")

    # Update the tools loading section to show status based on toggle
    tools = load_tool_configs() if st.session_state.function_calling_enabled else []
    if st.session_state.function_calling_enabled:
        if not tools:
            st.sidebar.warning("No tool configurations loaded")
        else:
            st.sidebar.success(f"Loaded {len(tools)} tools")

    available_models = get_ollama_models()
    model = st.sidebar.selectbox("Choose a model", available_models)
    
    # Add model health check
    if not asyncio.run(verify_model_health(model)):
        st.error(f"Error: Model '{model}' is not responding properly. Try pulling the model again using 'ollama pull {model}'")
        st.stop()
    
    logging.info(f"Model selected and verified: {model}")

    if model in st.session_state.model_messages and st.session_state.model_messages[model]:
        chat_history = export_chat_history(model, st.session_state.model_messages[model])
        if chat_history:
            st.sidebar.download_button(
                label="Download Chat History",
                data=chat_history,
                file_name=f"chat_history_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    if model not in st.session_state.model_messages:
        st.session_state.model_messages[model] = []

    current_messages = st.session_state.model_messages[model]

    for message in current_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your question"):
        with st.chat_message("user"):
            st.markdown(prompt)
        current_messages.append({"role": "user", "content": prompt})
        logging.info(f"User input: {prompt}")

        if current_messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                logging.info("Generating response")

                with st.spinner("Writing..."):
                    try:
                        if st.session_state.document_collection:
                            relevant_docs = query_documents(current_messages[-1]["content"],
                                                         st.session_state.document_collection)
                            if relevant_docs and relevant_docs['documents']:
                                context = "\nContext from documents:\n" + "\n".join(relevant_docs['documents'][0])
                                # Add context to the messages
                                current_messages.append({
                                    "role": "system",
                                    "content": f"Here's relevant context for the query: {context}"
                                })
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