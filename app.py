import streamlit as st
from llama_index.core.llms import ChatMessage
import logging
import time
from llama_index.llms.ollama import Ollama
import subprocess

logging.basicConfig(level=logging.INFO)

# Initialize session state for model-specific message histories
if 'model_messages' not in st.session_state:
    st.session_state.model_messages = {}

# Keep the existing messages state for backward compatibility
if 'messages' not in st.session_state:
    st.session_state.messages = []

def get_ollama_models():
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')[1:]
        models = [line.split()[0] for line in lines if line]
        return models
    except Exception as e:
        logging.error(f"Error getting Ollama models: {str(e)}")
        return ["llama3.1:latest"]

def stream_chat(model, messages):
    try:
        llm = Ollama(model=model, request_timeout=120.0)
        resp = llm.stream_chat(messages)
        response = ""
        response_placeholder = st.empty()
        for r in resp:
            response += r.delta
            response_placeholder.write(response)
        logging.info(f"Model: {model}, Messages: {messages}, Response: {response}")
        return response
    except Exception as e:
        logging.error(f"Error during streaming: {str(e)}")
        raise e

def main():
    st.title("Chat with LLMs Models")
    logging.info("App started")

    available_models = get_ollama_models()
    model = st.sidebar.selectbox("Choose a model", available_models)
    logging.info(f"Model selected: {model}")

    # Initialize message history for new models
    if model not in st.session_state.model_messages:
        st.session_state.model_messages[model] = []

    # Use the current model's message history
    current_messages = st.session_state.model_messages[model]

    # Display messages for the current model
    for message in current_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Get user input
    if prompt := st.chat_input("Your question"):
        with st.chat_message("user"):
            st.write(prompt)
        current_messages.append({"role": "user", "content": prompt})
        logging.info(f"User input: {prompt}")

        if current_messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                start_time = time.time()
                logging.info("Generating response")

                with st.spinner("Writing..."):
                    try:
                        messages = [ChatMessage(role=msg["role"], content=msg["content"])
                                  for msg in current_messages]
                        response_message = stream_chat(model, messages)
                        duration = time.time() - start_time
                        response_message_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
                        current_messages.append({
                            "role": "assistant",
                            "content": response_message_with_duration
                        })
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
