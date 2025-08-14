import streamlit as st
import logging
import asyncio
import time
import os
import glob

# Import our modules
from modules.session_manager import initialize_session_state, get_current_session, set_mcp_state
from modules.error_handler import display_error, ErrorCategory
from modules.model_manager import verify_model_health, stream_chat_with_tools, load_tool_configs, get_ollama_models
from tool_functions import available_functions  # re-export for tests patching
from modules.document_handler import query_documents
from modules.ui import apply_custom_css, render_sidebar, render_chat_history, render_model_selector, render_chat_download_button

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Apply custom CSS for styling
    apply_custom_css()
    
    st.title("Krytos AI Security Analyst")

    # Auto-connect to a local MCP server in mcp-servers/ if available and not already connected
    try:
        from modules import mcp_client
        if not st.session_state.get("mcp_connected", False) and mcp_client.is_available():
            servers_dir = os.path.join(os.getcwd(), "mcp-servers")
            candidates = []
            if os.path.isdir(servers_dir):
                candidates.extend(sorted(glob.glob(os.path.join(servers_dir, "*.py"))))
                candidates.extend(sorted(glob.glob(os.path.join(servers_dir, "*.js"))))
            if candidates:
                server_spec = candidates[0]
                try:
                    tools = asyncio.run(mcp_client.connect(server_spec))
                    set_mcp_state(True, server_spec, tools)
                    logging.info(f"Auto-connected to MCP server: {server_spec} with {len(tools)} tools")
                except Exception as e:
                    logging.warning(f"Auto-connect to MCP server failed: {e}")
    except Exception as e:
        logging.debug(f"MCP auto-connect skipped: {e}")
    
    # Render sidebar with document upload, settings, and session management
    render_sidebar()
    
    logging.info("App started")

    # Show tool status if function calling is enabled
    tools = load_tool_configs() if st.session_state.function_calling_enabled else []
    if st.session_state.function_calling_enabled:
        if not tools:
            st.sidebar.warning("No tool configurations loaded")
        else:
            st.sidebar.success(f"Loaded {len(tools)} tools")

    # Get current session data
    current_session = get_current_session()
    
    # Select model and update session with user's selection
    default_model_index = 0
    available_models = ["gpt-oss:20b"]  # Fallback value
    
    try:
        from modules.model_manager import get_ollama_models
        available_models = get_ollama_models()
        
        # Get the default model index - use the session's saved model if available
        if current_session['selected_model'] in available_models:
            default_model_index = available_models.index(current_session['selected_model'])
    except Exception as e:
        logging.error(f"Error getting models: {str(e)}")
        display_error(e, ErrorCategory.MODEL)
    
    # Render model selector
    model = render_model_selector(default_model_index)
    
    # Save the selected model with the current session
    current_session['selected_model'] = model
    
    # Add model health check
    if not asyncio.run(verify_model_health(model)):
        st.error(f"Error: Model '{model}' is not responding properly. Try pulling the model again using 'ollama pull {model}'")
        st.stop()
    
    logging.info(f"Model selected and verified: {model}")
    
    # Initialize model messages if needed
    if model not in current_session['model_messages']:
        current_session['model_messages'] = {model: []}
    
    # Add download button for chat history
    render_chat_download_button(model, current_session['model_messages'].get(model, []))

    # Get current messages for selected model
    current_messages = current_session['model_messages'].get(model, [])
    
    # Display current dossier info
    st.info(f"Current Dossier: {current_session['name']}")

    # Render chat history
    render_chat_history(current_messages)

    # Handle user input
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
                        response_message = asyncio.run(stream_chat_with_tools(model, current_messages))
                        
                        # Calculate duration for display
                        duration = time.time() - start_time
                        
                        # Format the full response to display to the user
                        display_message = response_message
                        
                        # Add the complete response to the message history without tool details
                        current_messages.append({
                            "role": "assistant",
                            "content": response_message
                        })
                        
                        # Display the response and duration
                        st.markdown(display_message)
                        # Tool details are already summarized in the response when applicable
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