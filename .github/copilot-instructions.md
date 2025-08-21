# Copilot instructions for this repo

Purpose: Streamlit app that chats with Ollama, can call security tools (port scan, domain info, vuln checks, SQLi simulation), and optionally ground answers with PDF docs stored/searched via Milvus.

## Architecture map
- `app.py`: Streamlit entry; wires sidebar, chat loop, doc context, tool status.
- `modules/ui.py`: Sidebar (upload, settings, sessions), chat history rendering, model selector, chat export.
- `modules/session_manager.py`: “Dossier” sessions; keys: `sessions`, `current_session_id`, `model_messages`, `document_collection`, `selected_model`, `function_calling_enabled`, `milvus_connected`, `milvus_lite_db_file`.
- `modules/model_manager.py`: `get_ollama_models`, `verify_model_health`, `stream_chat_with_tools`, `export_chat_history`, `load_tool_configs` (reads `tools_config.yaml`).
- `modules/error_handler.py`: `display_error(exc, ErrorCategory.*)` for user-visible failures.
- `modules/document_handler.py`: Upload flow, Milvus connect UI, `query_documents()` (read path via `pymilvus.Collection`).
- `document_loader.py`: PDF→chunks (LangChain), embeddings (sentence-transformers), Milvus writes via `MilvusClient`.
- Tool runtime: `async_tools.py` (processes tool_calls and re-queries model), `tool_functions.py` (tool implementations), `tools_config.yaml` (schemas).

## Chat + tools flow
1) Append user prompt to `current_session['model_messages'][model]`.
2) Call `model_manager.stream_chat_with_tools()`; pass tools only if `function_calling_enabled`.
3) If model returns tool_calls, `async_tools.process_tool_calls()` dispatches to `tool_functions.available_functions[name](**args)` and appends `{role:'tool', name, content}`.
4) Call model again with updated messages → return final assistant text; UI shows “Tool Usage Details” separately.

## Conventions to follow
- Errors: Never show raw exceptions; call `display_error(e, ErrorCategory.XXX)`. Prefer specific exceptions (`EmbeddingError`, `MilvusConnectionError`, `NetworkToolError`, etc.).
- Tools: Return dicts with `timestamp` and `status`. For failures use `create_error_response(tool_name, e, target)`.
- Add a tool: implement in `tool_functions.py`, register in `available_functions`, add YAML schema in `tools_config.yaml` with matching arg names.
- Messages: Keep `{'role','content'}` (optional `name`). Filter before model calls.
- Models: Discover with `ollama list` (fallback `llama3.1:latest`). Sessions default to `digiguru/krytos:latest`. Always `verify_model_health(model)`.
- Milvus: Writes via `DocumentLoader.setup_milvus()` (vector field `embedding`, COSINE; `content` VARCHAR; `metadata` JSON). Reads via `modules/document_handler.query_documents()`.

## Dev workflows
- Run app: `streamlit run app.py` (ensure `ollama serve`; pull a model).
- Tests: `python run_tests.py`.
  - Unit: `--unit`; Integration: `--integration`; All+coverage: `--all --coverage --html`.
  - UI tests are skipped unless `--ui` or `--all`.

## Integration notes
- Nmap may need elevated perms; errors mapped to user-friendly messages in `tool_functions.py`.
- Embeddings use `sentence-transformers/all-MiniLM-L6-v2`; `DocumentLoader(testing_mode=True)` fakes embeddings/chunks.
- Milvus Lite DB default `milvus_app.db`; if bare filename, code prepends `./`.
- Chat export: `export_chat_history()` → sidebar download in `ui.render_chat_download_button`.

Questions or gaps (tell me and I’ll refine): e.g., adding new sidebar settings, extending Milvus schema, or supporting additional model providers.
