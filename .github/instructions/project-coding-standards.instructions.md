---
applyTo: "**/*.py"
---
# Project Coding Standards for Python

Apply the [general coding guidelines](./general-coding.instructions.md) to all code.

## Python Guidelines

- Use snake_case for functions/variables and PascalCase for classes and Exceptions.
- Add type hints for public functions and major internal helpers.
- Keep comments minimal; prefer `logging` for runtime context over inline comments.
- Group imports: stdlib, third-party, local; avoid unused imports.
- Raise specific exceptions (e.g., `EmbeddingError`, `MilvusConnectionError`) rather than generic ones; surface errors via `modules.error_handler.display_error`.

## Streamlit App & Session State

- Centralize session state in `modules/session_manager.py`; use these keys consistently:
  - `model_messages`, `document_collection`, `milvus_connected`, `milvus_lite_db_file`,
    `function_calling_enabled`, `sessions`, `current_session_id`, `selected_model`
- UI code in `modules/ui.py` should be presentational; invoke business logic from modules.
- After switching sessions, call `st.rerun()` to refresh UI.
- Inject CSS via `apply_custom_css()`; avoid inline styles elsewhere.

## Model I/O and Tool Calling (Ollama)

- Use `ollama.AsyncClient().chat` for model calls.
- When tool-calling is enabled:
  - Load tools from `tools_config.yaml` via `modules/model_manager.load_tool_configs()`.
  - Map tool names to callables in `tool_functions.available_functions`. Names must match YAML.
  - On tool calls, append `{'role': 'tool', 'content': str(output), 'name': tool_name}` to messages, then make a second model call.
- Always send only `{'role', 'content'}` pairs to the model (filter out extra fields before calling).

Example message shape:
- `{"role": "user", "content": "Scan example.com"}`
- `{"role": "tool", "name": "scan_network", "content": "{\"status\": \"completed\", ...}"}`

## Documents, Embeddings, and Milvus

- PDFs: `PyPDFLoader` + `RecursiveCharacterTextSplitter` with `chunk_size=1000`, `chunk_overlap=100`.
- Embeddings: `from langchain_huggingface import HuggingFaceEmbeddings` with model `sentence-transformers/all-MiniLM-L6-v2`.
  - Ensure `torch` and `sentence-transformers` are installed and importable.
  - In `DocumentLoader.get_embeddings`, raise `EmbeddingError` with helpful messages (“model not found”, OOM).
- Milvus:
  - Writes: Prefer `pymilvus.MilvusClient` (see `DocumentLoader.setup_milvus`).
    - Row format: `{"content": str, "metadata": json.dumps(dict), "embedding": List[float]}`
    - Vector field: `embedding` with COSINE metric; `content` as `VARCHAR`, `metadata` as `JSON`.
    - Call `flush` after inserts.
  - Reads: UI path uses `pymilvus.connections.connect` + `Collection.search`; check existence via `utility.has_collection`.
- Keep the dual path (MilvusClient for writes, Collection for reads) consistent; don’t mix APIs within a single function.

## Error Handling and UX

- Do not surface raw exceptions in the UI. Always call:
  - `display_error(exc, ErrorCategory.XXX)` from `modules.error_handler`.
- Map errors to categories: `CONNECTION`, `MODEL`, `DOCUMENT`, `TOOL`, `MILVUS`, `UNKNOWN`.
- Tool errors should return a JSON-serializable dict via `create_error_response(...)`.

## Logging

- Use `logging.info` for flow milestones (start, success), `logging.error` for failures with `str(e)`.
- Include actionable context (identifiers, collection names, model names) in logs; avoid sensitive data.

## Asynchronous Patterns

- Prefer `async` for model calls and tool post-processing (`async_tools.process_tool_calls`).
- Avoid blocking I/O in async paths; if necessary, isolate in sync helpers and call appropriately.

## Testing

- Use `run_tests.py` to run tests and coverage:
  - `python run_tests.py --unit`, `--integration`, `--all --coverage --html`
  - UI tests are skipped unless `--ui` is passed (via `SKIP_UI_TESTS`).
- Patch external services (Ollama, subprocess, Milvus, network tools) using `unittest.mock.patch`.
- For async, use `AsyncMock` (see tests) or `pytest-asyncio`’s patterns.

## Configuration & Tools

- Tool schemas live in `tools_config.yaml`; keep parameter names/types aligned with function signatures.
- Register tools in `tool_functions.available_functions` with exact YAML names.
- When editing embeddings or Milvus logic, update tests and ensure dependencies in `requirements.txt` cover `langchain-huggingface`, `torch`, and `sentence-transformers`.
