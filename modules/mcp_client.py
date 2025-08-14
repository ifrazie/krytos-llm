import logging
import shlex
import importlib
from typing import Any, Dict, List, Optional, Tuple


# Internal state (module-level, managed via async connect/disconnect)
_exit_stack: Any = None  # AsyncExitStack at runtime
_session: Any = None  # ClientSession at runtime
_tools_cache: List[Dict[str, Any]] = []
_transport_info: Optional[Tuple[str, List[str]]] = None  # (command, args)


def is_available() -> bool:
    """Return True if the MCP Python package is importable."""
    try:
        importlib.import_module("mcp")
        importlib.import_module("mcp.client.stdio")
        return True
    except Exception:
        return False


def is_connected() -> bool:
    """Return True if there's an active MCP session."""
    return _session is not None


def _infer_command_and_args(server_spec: str) -> Tuple[str, List[str]]:
    """
    Infer the command and args to launch an MCP server over stdio.
    Accepts a file path (.py/.js) or a full shell command (e.g., "npx -y @modelcontextprotocol/server-brave-search").
    """
    server_spec = server_spec.strip()
    if server_spec.endswith(".py"):
        # Prefer "python" for cross-platform; user may rely on PATH
        return "python", [server_spec]
    if server_spec.endswith(".js"):
        return "node", [server_spec]
    # Treat as a shell command
    parts = shlex.split(server_spec)
    if not parts:
        raise ValueError("Empty MCP server command or path")
    return parts[0], parts[1:]


async def connect(server_spec: str, env: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """
    Connect to an MCP server via stdio and cache available tools.

    Args:
        server_spec: Path to server script (.py/.js) or a shell command string.
        env: Optional environment variables for the subprocess.

    Returns:
        List of tool descriptors as returned by the MCP protocol.
    """
    global _exit_stack, _session, _tools_cache, _transport_info
    if not is_available():
        raise RuntimeError("MCP Python package is not installed. Install 'mcp' to enable MCP client features.")

    if _session is not None:
        # Already connected; return cached tools
        return _tools_cache

    command, args = _infer_command_and_args(server_spec)
    _transport_info = (command, args)

    # Import MCP types only when needed via importlib to avoid static import errors
    AsyncExitStack = importlib.import_module("contextlib").AsyncExitStack  # type: ignore[attr-defined]
    mcp_mod = importlib.import_module("mcp")
    stdio_mod = importlib.import_module("mcp.client.stdio")
    ClientSession = getattr(mcp_mod, "ClientSession")
    StdioServerParameters = getattr(mcp_mod, "StdioServerParameters")
    stdio_client = getattr(stdio_mod, "stdio_client")

    _exit_stack = AsyncExitStack()
    server_params = StdioServerParameters(command=command, args=args, env=env)

    stdio, write = await _exit_stack.enter_async_context(stdio_client(server_params))
    _session = await _exit_stack.enter_async_context(ClientSession(stdio, write))
    await _session.initialize()

    # Fetch tools and cache
    tools_response = await _session.list_tools()
    _tools_cache = []
    for t in tools_response.tools:
        try:
            _tools_cache.append({
                "name": getattr(t, "name", ""),
                "description": getattr(t, "description", ""),
                "input_schema": getattr(t, "inputSchema", {}),
            })
        except Exception as e:
            logging.error(f"Failed to parse MCP tool entry: {e}")
    logging.info(f"Connected to MCP server with tools: {[t['name'] for t in _tools_cache]}")
    return _tools_cache


async def disconnect() -> None:
    """Disconnect from the MCP server and cleanup resources."""
    global _exit_stack, _session, _tools_cache, _transport_info
    if _exit_stack is not None:
        try:
            await _exit_stack.aclose()
        except Exception as e:  # pragma: no cover
            logging.warning(f"Error while closing MCP client: {e}")
    _exit_stack = None
    _session = None
    _tools_cache = []
    _transport_info = None


async def list_tools() -> List[Dict[str, Any]]:
    """Return the cached tools; refresh from server if available and connected."""
    if _session is None:
        return []
    try:
        tools_response = await _session.list_tools()
        tools: List[Dict[str, Any]] = []
        for t in tools_response.tools:
            tools.append({
                "name": getattr(t, "name", ""),
                "description": getattr(t, "description", ""),
                "input_schema": getattr(t, "inputSchema", {}),
            })
        # update cache
        global _tools_cache
        _tools_cache = tools
        return tools
    except Exception as e:
        logging.error(f"Failed to list MCP tools: {e}")
        return _tools_cache


def get_tool_names() -> List[str]:
    """Return the names of currently cached MCP tools."""
    return [t.get("name", "") for t in _tools_cache]


def get_ollama_tool_specs() -> List[Dict[str, Any]]:
    """
    Convert cached MCP tools to Ollama-compatible tool definitions for function calling.
    Returns a list of {"type":"function","function":{name,description,parameters}} entries.
    """
    specs: List[Dict[str, Any]] = []
    for t in _tools_cache:
        name = t.get("name")
        desc = t.get("description") or ""
        schema = t.get("input_schema") or {}
        # Normalize schema to an object per Ollama/OpenAI tools format
        parameters: Dict[str, Any] = {
            "type": "object",
        }
        if isinstance(schema, dict):
            # Carry over common fields if present
            if "type" in schema:
                parameters["type"] = schema.get("type", "object")
            if "properties" in schema:
                parameters["properties"] = schema.get("properties", {})
            if "required" in schema:
                parameters["required"] = schema.get("required", [])
        specs.append({
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": parameters,
            }
        })
    return specs


async def call_tool(name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Call an MCP tool by name with given arguments. Returns a JSON-serializable dict.
    The returned structure normalizes MCP content into a simple dict for display/logging.
    """
    if _session is None:
        raise RuntimeError("MCP client is not connected.")
    arguments = arguments or {}
    try:
        result = await _session.call_tool(name, arguments)
        # Normalize result.content which may be a list of content parts
        content = getattr(result, "content", None)
        # Best-effort to render content to plain text
        text_chunks: List[str] = []
        try:
            for c in content or []:
                ctype = getattr(c, "type", None)
                if ctype == "text":
                    text_chunks.append(getattr(c, "text", ""))
                else:
                    # Fallback to string representation
                    text_chunks.append(str(c))
        except Exception:  # pragma: no cover
            text_chunks = [str(content)] if content is not None else []
        return {
            "status": "completed",
            "tool": name,
            "content": "\n".join([t for t in text_chunks if t]),
            "raw": None if content is None else str(content),
        }
    except Exception as e:
        logging.error(f"MCP tool call failed for {name}: {e}")
        return {
            "status": "error",
            "tool": name,
            "error": str(e),
        }
