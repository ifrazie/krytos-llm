import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Import the functions to test
from tool_functions import available_functions
from app import load_tool_configs, stream_chat_with_tools, get_ollama_models, verify_model_health

class TestAppIntegration:
    """Test the integration between app components"""
    
    def test_load_tool_configs(self):
        """Test loading tool configurations from YAML"""
        # Call the function
        tools = load_tool_configs()
        
        # Assertions
        assert isinstance(tools, list)
        assert len(tools) > 0
        assert all("type" in tool for tool in tools)
        assert all("function" in tool for tool in tools)
        assert all(tool["type"] == "function" for tool in tools)
    
    @pytest.mark.asyncio
    @patch('ollama.AsyncClient')
    async def test_verify_model_health(self, mock_client_class):
        """Test model health verification"""
        # Setup mock client
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.return_value = MagicMock()
        
        # Call the function
        result = await verify_model_health("llama3.1:latest")
        
        # Assertions
        mock_client_class.assert_called_once()
        mock_client.chat.assert_called_once()
        assert result is True
        
        # Test failure scenario
        mock_client.chat.side_effect = Exception("Model not found")
        result = await verify_model_health("nonexistent-model")
        assert result is False
    
    @patch('subprocess.run')
    def test_get_ollama_models(self, mock_run):
        """Test getting available Ollama models"""
        # Setup mock
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.stdout = "NAME                    ID              SIZE    MODIFIED\nllama3.1:latest        12345abcde      5.0GB   2 weeks ago\nllama2:13b             67890fghij      13.0GB  1 month ago"
        mock_run.return_value = mock_process
        
        # Call the function
        models = get_ollama_models()
        
        # Assertions
        mock_run.assert_called_once()
        assert "llama3.1:latest" in models
        assert "llama2:13b" in models
        assert len(models) == 2
        
        # Test failure scenario
        mock_process.returncode = 1
        mock_run.return_value = mock_process
        models = get_ollama_models()
        assert models == ["llama3.1:latest"]  # Default fallback
    
    @pytest.mark.asyncio
    @patch('ollama.AsyncClient')
    @patch('app.load_tool_configs')
    @patch('app.available_functions')
    async def test_stream_chat_with_tools(self, mock_available_functions, mock_load_tool_configs, mock_client_class):
        """Test streaming chat with tool calls"""
        # Setup mocks
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # First response with tool calls
        first_response = MagicMock()
        first_response.message.content = "Let me check that for you"
        
        # Create mock tool call
        tool_call = MagicMock()
        tool_call.function.name = "get_info"
        tool_call.function.arguments = {"domain": "example.com"}
        first_response.message.tool_calls = [tool_call]
        first_response.message.role = "assistant"
        
        # Final response after tool call
        final_response = MagicMock()
        final_response.message.content = "Here are the results"
        final_response.message.tool_calls = None
        
        # Configure mock client to return the responses
        mock_client.chat.side_effect = [first_response, final_response]
        
        # Configure mock tool function
        mock_tool_function = MagicMock(return_value={"status": "completed", "domain": "example.com"})
        mock_available_functions.get.return_value = mock_tool_function
        
        # Configure mock tool configs
        mock_load_tool_configs.return_value = [{"type": "function", "function": {"name": "get_info"}}]
        
        # Call the function with function calling enabled
        messages = [{"role": "user", "content": "Get info on example.com"}]
        
        # Set session state mock
        with patch('app.st.session_state') as mock_session_state:
            mock_session_state.function_calling_enabled = True
            
            result = await stream_chat_with_tools("llama3.1:latest", messages)
            
        # Assertions
        assert mock_client.chat.call_count == 2
        mock_available_functions.get.assert_called_once_with("get_info")
        mock_tool_function.assert_called_once_with(domain="example.com")
        assert result == "Here are the results"