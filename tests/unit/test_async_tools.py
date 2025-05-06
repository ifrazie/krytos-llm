import pytest
import asyncio
from unittest.mock import patch, MagicMock

from async_tools import process_tool_calls

class TestAsyncTools:
    """Test the async_tools module"""
    
    @pytest.mark.asyncio
    @patch('async_tools.available_functions')
    async def test_process_tool_calls_with_tools(self, mock_available_functions):
        """Test processing tool calls with tool responses"""
        # Setup mocks
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=MagicMock(message=MagicMock(content="Final response")))
        
        mock_function = MagicMock(return_value={"result": "tool_output"})
        mock_available_functions.get.return_value = mock_function
        
        # Create mock response with tool calls
        mock_response = MagicMock()
        mock_tool_call = MagicMock()
        mock_tool_call.function.name = "test_function"
        mock_tool_call.function.arguments = {"param1": "value1"}
        mock_response.message.tool_calls = [mock_tool_call]
        mock_response.message.content = "Tool call response"
        mock_response.message.role = "assistant"
        
        # Call the function
        result = await process_tool_calls(
            mock_client,
            "test_model",
            [{"role": "user", "content": "test message"}],
            mock_response
        )
        
        # Assertions
        mock_available_functions.get.assert_called_once_with("test_function")
        mock_function.assert_called_once_with(param1="value1")
        mock_client.chat.assert_called_once()
        assert "Final response" in result
        
    @pytest.mark.asyncio
    async def test_process_tool_calls_without_tools(self):
        """Test processing response without tool calls"""
        # Setup mocks
        mock_client = MagicMock()
        
        # Create mock response without tool calls
        mock_response = MagicMock()
        mock_response.message.tool_calls = None
        mock_response.message.content = "Simple response"
        
        # Call the function
        result = await process_tool_calls(
            mock_client,
            "test_model",
            [{"role": "user", "content": "test message"}],
            mock_response
        )
        
        # Assertions
        assert result == "Simple response"
        assert not mock_client.chat.called

# Helper class for mocking async methods
class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)