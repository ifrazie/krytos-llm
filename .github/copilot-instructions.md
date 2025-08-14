# AI assistant guide for this repo (krytos-llm)

This project is a Streamlit app that chats with Ollama LLMs, can call security tools, and performs RAG over PDFs stored in Milvus.

## Big picture
- UI and app flow: `app.py` drives Streamlit; UI helpers live in `modules/ui.py`.
- Session state: `modules/session_manager.py` owns Streamlit `st.session_state` keys (model, messages, dossier, Milvus connection, function-calling toggle).
- Model I/O and tools: `modules/model_manager.py` talks to Ollama (list models, health check, chat). If function calling is enabled, it loads tools from `tools_config.yaml` and passes them to the model. Tool execution + second-round chat happens in `async_tools.py` using `tool_functions.py`.
- Documents/RAG: Upload + Milvus connection in `modules/document_handler.py`; PDF loading, chunking, embeddings, and Milvus CRUD live in `document_loader.py`.
- Errors: Centralized formatting and display via `modules/error_handler.py`.
- Config/tests: `tools_config.yaml` defines tool schemas; tests in `tests/**`; runner `run_tests.py`.

## End-to-end chat + tools flow
1. User enters a prompt in `app.py`.
2. If a document collection is set, `modules/document_handler.query_documents` computes a query embedding via `DocumentLoader.get_embeddings()` and searches Milvus, adding retrieved context as a system message.
3. `modules/model_manager.stream_chat_with_tools` calls `ollama.AsyncClient().chat` with messages and optional tools (from YAML) when `st.session_state.function_calling_enabled` is True.
4. If the model returns tool calls, `async_tools.process_tool_calls` executes concrete functions from `tool_functions.available_functions`, appends each result as a `{'role': 'tool', 'content': str(output), 'name': tool_name}` message, then calls the model again to get the final response.

## Documents and embeddings
- PDF loading: `document_loader.py` uses `PyPDFLoader` and `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=100).
- Embeddings: `langchain_huggingface.HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2` is used; ensure `torch` and `sentence-transformers` are installed and importable.
- Milvus: Two paths are used in the codebase:
  - UI connection uses `pymilvus.connections.connect` (Lite via local DB file or Standalone via host/port) and the low-level `Collection` API for querying.
  - `DocumentLoader` uses the high-level `pymilvus.MilvusClient` for collection creation/inserts and also exposes `query_documents` (UI path currently uses the module-level `document_handler.query_documents`).
- Collection schema (via `MilvusClient`): vector field `embedding` (COSINE metric), plus `content` (VARCHAR) and `metadata` (JSON). Inserts are row-oriented: `{content, metadata(json), embedding}`.

## Adding or changing tools (function calling)
- Define the tool schema in `tools_config.yaml` (name, description, JSON schema for parameters).
- Implement the function in `tool_functions.py` and register it in `available_functions` (keys must match the YAML `name`).
- Functions should return JSON-serializable dicts and raise specific exceptions when appropriate; errors are normalized via `create_error_response()`.
- Example: `get_info(domain: str) -> dict`, `scan_network(ip_address: str) -> dict`, `check_vulnerability(domain: str) -> dict`, `sql_injection(url: str) -> dict`.

## Conventions and patterns
- Messages are simple dicts: `{role: 'user'|'assistant'|'system'|'tool', content: str, name?: str}`.
- Tool call results are added as separate `role: 'tool'` messages before the second model call.
- Session keys you’ll see: `model_messages`, `document_collection`, `milvus_connected`, `milvus_lite_db_file`, `function_calling_enabled`, `sessions`, `current_session_id`.
- Errors: don’t surface raw exceptions to UI—log and let `modules/error_handler.display_error` format them. It maps common phrases ("model not found", "embedding", "search", etc.) to user-friendly messages.

## Critical workflows
- Run the app:
  ```bash
  streamlit run app.py
  ```
  Prereqs: Ollama running (`ollama serve`) and at least one model pulled (e.g., `ollama pull llama3.1:latest`).
- Connect Milvus in the sidebar:
  - Milvus Lite: provide a DB filename (e.g., `milvus_app.db`); code prepends `./` if needed.
  - Standalone: set host/port.
- Run tests (UI tests are skipped unless `--ui`):
  ```bash
  python run_tests.py --unit
  python run_tests.py --integration
  python run_tests.py --all --coverage --html
  ```

## External deps and gotchas
- Ollama: `modules/model_manager.get_ollama_models()` shells to `ollama list`; failures fall back to `llama3.1:latest`.
- Embeddings: this repo imports from `langchain_huggingface`; ensure the package is installed alongside `torch` and `sentence-transformers`.
- Milvus: Lite uses a local file path; Standalone uses `http://{host}:{port}`. Ensure the collection exists before querying (`document_handler.query_documents` checks via `utility.has_collection`).
- Network tools may require privileges (e.g., Nmap); code raises `NetworkToolError` with a helpful message.

## Where to look when changing X
- UI/UX: `modules/ui.py`
- Model and function-calling behavior: `modules/model_manager.py`, `async_tools.py`, `tools_config.yaml`, `tool_functions.py`
- Docs & RAG: `modules/document_handler.py`, `document_loader.py`
- Errors and UX messaging: `modules/error_handler.py`
- Tests and runner: `tests/**`, `run_tests.py`
