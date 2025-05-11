# Krytos AI

AI-powered chat interface for security analysis that integrates with Ollama LLMs and provides security-focused function calling capabilities.

![Krytos AI](images/Screenshot%202025-03-15%20003037.png)

## Features

- Interactive chat interface with Ollama large language models
- Document retrieval using Milvus vector database
- Security-focused function calling capabilities:
  - Domain information gathering
  - Network port scanning
  - Vulnerability assessment
  - SQL injection testing
- Session management for organizing conversations
- Document upload and analysis

## Installation

### Prerequisites

- Python 3.9+ 
- [Ollama](https://github.com/ollama/ollama) installed and running
- (Optional) Milvus database for document storage

### Installation Steps

1. **Clone the repository:**

```bash
git clone https://github.com/ifrazie/krytos-llm.git
cd krytos-llm
```

2. **Create a virtual environment:**

```bash
python -m venv venv
```

3. **Activate the virtual environment:**

- On Windows:

```bash
venv\Scripts\activate
```

- On macOS/Linux:

```bash
source venv/bin/activate
```

4. **Install dependencies:**

```bash
pip install -r requirements.txt
```

5. **Download at least one Ollama model:**

```bash
ollama pull llama3.2:latest
```

## Usage

### Starting the Application

1. **Ensure Ollama is running.**

2. **Start the Streamlit application:**

```bash
streamlit run app.py
```

3. **Access the application in your browser:** http://localhost:8501

### Using the Application

#### Chat Interface

1. Select a model from the dropdown in the sidebar.
2. Type your questions or commands in the chat input.
3. For security analysis, enable function calling in the settings panel.

#### Document Analysis

1. Upload PDF documents using the uploader in the sidebar.
2. Connect to a Milvus database (either Milvus Lite or standalone).
3. Ask questions about the documents to retrieve relevant information.

#### Session Management

1. Create new sessions ("Dossiers") for different analysis tasks.
2. Switch between sessions using the dropdown in the sidebar.
3. Download chat history for archiving or sharing.

### Example Commands

- **Domain Information:** "Get information about example.com"
- **Port Scanning:** "Scan 192.168.1.1 for open ports"
- **Vulnerability Assessment:** "Check for vulnerabilities on example.com"
- **SQL Injection Testing:** "Test https://example.com/login for SQL injection"

## Development

### Project Structure

```
streamlit-ollama-chat/
├── app.py                  # Main application entry point
├── async_tools.py          # Asynchronous tool handling
├── document_loader.py      # Document processing functionality
├── tool_functions.py       # Security tool implementations
├── tools_config.yaml       # Tool configurations
├── modules/                # Core application modules
│   ├── document_handler.py # Document upload and query
│   ├── error_handler.py    # Error handling and display
│   ├── model_manager.py    # LLM model management
│   ├── session_manager.py  # Session state management
│   └── ui.py               # User interface components
└── tests/                  # Test suite
    ├── integration/        # Integration tests
    ├── unit/               # Unit tests
    └── ui/                 # UI tests
```

### Running Tests

The project includes a comprehensive test suite:

```bash
# Run all tests
python run_tests.py --all

# Run only unit tests
python run_tests.py --unit

# Run with coverage
python run_tests.py --coverage

# Generate HTML coverage report
python run_tests.py --coverage --html
```

### Adding New Tools

1. Define the tool in `tools_config.yaml` with parameters.
2. Implement the tool function in `tool_functions.py`.
3. Register the function in the `available_functions` dictionary.

Example:

```yaml
# in tools_config.yaml
- name: new_tool
  description: Description of the new tool
  parameters:
    type: object
    required: [ param1 ]
    properties:
      param1:
        type: string
        description: Description of parameter
```

```python
# in tool_functions.py
def new_tool(param1: str) -> dict:
    """Implementation of the new tool"""
    # Tool implementation
    return {"result": "output"}

# Update the available_functions dictionary
available_functions = {
    # ...existing tools...
    'new_tool': new_tool
}
```

## Troubleshooting

- **Ollama Connection Issues:** Ensure Ollama is running (`ollama serve`).
- **Model Not Found:** Pull the model first (`ollama pull model_name`).
- **Milvus Connection:** Check Milvus connection settings in the sidebar.
- **Function Calling Issues:** Ensure the model supports function calling.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.

This license allows for both open source and proprietary use, but requires attribution in all cases.

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [Ollama](https://ollama.ai/)
- [Milvus](https://milvus.io/)
- [LangChain](https://www.langchain.com/)