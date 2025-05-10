import logging
import yaml
import subprocess
import asyncio
import ollama
from typing import List, Dict, Any, Tuple, Optional

from modules.error_handler import display_error, ErrorCategory
from async_tools import process_tool_calls, format_tool_info_for_display

def get_ollama_models() -> List[str]:
    """
    Get list of available Ollama models
    
    Returns:
        List[str]: List of available model names
    """
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

async def verify_model_health(model_name: str) -> bool:
    """
    Verify if the model is properly loaded and functioning.
    
    Args:
        model_name: The name of the model to check
        
    Returns:
        bool: True if model is healthy, False otherwise
    """
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

def load_tool_configs() -> List[Dict[str, Any]]:
    """
    Load tool configurations from YAML file
    
    Returns:
        List[Dict[str, Any]]: List of tool configurations
    """
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

async def stream_chat_with_tools(model: str, messages: List[Dict[str, Any]]) -> Tuple[str, Optional[List[str]]]:
    """
    Process a chat with the model, potentially using tools
    
    Args:
        model: The model name to use
        messages: List of messages in the conversation history
        
    Returns:
        Tuple[str, Optional[List[str]]]: The model's response and optional tool info
    """
    import streamlit as st
    
    try:
        client = ollama.AsyncClient()
        tools = load_tool_configs() if st.session_state.function_calling_enabled else None

        response = await client.chat(
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

def export_chat_history(model: str, messages: List[Dict[str, Any]]) -> Optional[str]:
    """
    Export chat history to JSON
    
    Args:
        model: The model used for the conversation
        messages: The conversation message history
        
    Returns:
        Optional[str]: JSON string or None if no messages
    """
    import json
    from datetime import datetime
    from modules.session_manager import serialize_message
    
    if not messages:
        return None

    chat_data = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "messages": [serialize_message(msg) for msg in messages]
    }

    return json.dumps(chat_data, indent=2)