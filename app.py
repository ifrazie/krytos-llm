import streamlit as st
import logging
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

logging.basicConfig(level=logging.INFO)

# Initialize session state
if 'model_messages' not in st.session_state:
    st.session_state.model_messages = {}

if 'messages' not in st.session_state:
    st.session_state.messages = []

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

def get_ollama_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')[1:]
        models = [line.split()[0] for line in lines if line]
        return models
    except Exception as e:
        logging.error(f"Error getting Ollama models: {str(e)}")
        return ["llama3.1:latest"]

async def stream_chat_with_tools(model, messages):
    tool = None
    final_response = None
    try:
        client = ollama.AsyncClient()
        tools = load_tool_configs()

        response: ChatResponse = await client.chat(
            model,
            messages=[{"role": m["role"], "content": m["content"]} for m in messages],
            tools=tools
        )

        tool_output = None
        tool_info = []

        if response.message.tool_calls:
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
            messages.append(response.message)
            messages.append({'role': 'tool', 'content': str(tool_output), 'name': tool.function.name})
            final_response = await client.chat(model, messages=[{"role": m["role"], "content": m["content"]} for m in messages])

        # Combine the final response with tool information
        complete_response = final_response.message.content
        if tool_info:
            complete_response += "\n\n💡 Tool Usage Details:" + "".join(tool_info)

            return complete_response

        return response.message.content
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e

def export_chat_history(model, messages):
    if not messages:
        return None

    chat_data = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "messages": messages
    }

    return json.dumps(chat_data, indent=2)

def main():
    st.title("Chat with LLMs Models")
    logging.info("App started")

    tools = load_tool_configs()
    if not tools:
        st.sidebar.warning("No tool configurations loaded")
    else:
        st.sidebar.success(f"Loaded {len(tools)} tools")

    available_models = get_ollama_models()
    model = st.sidebar.selectbox("Choose a model", available_models)
    logging.info(f"Model selected: {model}")

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
