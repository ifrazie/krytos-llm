from ollama import ChatResponse
from typing import Dict, Any, List, Optional, Tuple
import logging

# Import the functions and dictionary from tool_functions.py
from tool_functions import available_functions

class ToolExecutionError(Exception):
    """Exception raised when a tool fails to execute properly"""
    def __init__(self, tool_name: str, error_message: str):
        self.tool_name = tool_name
        self.error_message = error_message
        super().__init__(f"Error executing tool '{tool_name}': {error_message}")

class ToolResponseError(Exception):
    """Exception raised when the model's response after tool call fails"""
    def __init__(self, error_message: str):
        self.error_message = error_message
        super().__init__(f"Error getting final response after tool execution: {error_message}")

async def process_tool_calls(client, model: str, messages: List[Dict[str, Any]], 
                            response: ChatResponse) -> Tuple[str, Optional[List[Dict[str, Any]]]]:
    """
    Process tool calls from the model response and get the final response.
    
    Args:
        client: The Ollama client
        model: The model name
        messages: The message history
        response: The model's response containing tool calls
        
    Returns:
        Tuple[str, Optional[List[Dict[str, Any]]]]: A tuple containing:
            - The final response after processing tool calls
            - Optional list of tool usage details for display
    """
    tool_outputs = []
    tool_info = []
    
    # Process each tool call
    if response.message.tool_calls:
        for tool in response.message.tool_calls:
            tool_name = tool.function.name
            tool_args = tool.function.arguments
            
            if function_to_call := available_functions.get(tool_name):
                # Create tool call info string
                tool_call_info = (
                    f"\n🔧 Tool Called: {tool_name}\n"
                    f"📝 Parameters: {tool_args}\n"
                )
                
                try:
                    # Call the function with the provided arguments
                    logging.info(f"Executing tool: {tool_name} with args: {tool_args}")
                    tool_output = function_to_call(**tool_args)
                    
                    # Store the output for adding to messages
                    tool_outputs.append({
                        'output': tool_output,
                        'name': tool_name
                    })
                    
                    # Add result to tool info for display
                    tool_call_info += f"📊 Result: {tool_output}\n"
                    
                except Exception as e:
                    error_message = str(e)
                    logging.error(f"Tool execution error in {tool_name}: {error_message}")
                    
                    # Create a user-friendly error message
                    if "missing" in error_message.lower() and "required" in error_message.lower():
                        user_message = f"The tool '{tool_name}' is missing required parameters."
                    elif "permission" in error_message.lower():
                        user_message = f"The tool '{tool_name}' doesn't have permission to execute."
                    elif "not found" in error_message.lower():
                        user_message = f"A resource needed by the tool '{tool_name}' was not found."
                    elif "timeout" in error_message.lower():
                        user_message = f"The tool '{tool_name}' timed out during execution."
                    else:
                        user_message = f"The tool '{tool_name}' encountered an error during execution."
                    
                    # Add error information to tool info
                    tool_call_info += f"❌ Error: {user_message}\n"
                    tool_call_info += f"Details: {error_message}\n"
                    
                    # Create an error result to pass to the model
                    error_output = {
                        "status": "error",
                        "error": user_message,
                        "details": error_message
                    }
                    
                    tool_outputs.append({
                        'output': error_output,
                        'name': tool_name
                    })
                
                tool_info.append(tool_call_info)
            else:
                logging.warning(f"Unknown tool called: {tool_name}")
                tool_call_info = (
                    f"\n❓ Unknown Tool Called: {tool_name}\n"
                    f"📝 Parameters: {tool_args}\n"
                    f"❌ Error: Tool not found in available tools\n"
                )
                tool_info.append(tool_call_info)
                
                # Add an error message for unknown tools
                tool_outputs.append({
                    'output': {"status": "error", "error": f"Tool '{tool_name}' is not available."},
                    'name': tool_name
                })
    
    # If we have tool outputs, get final response from model
    if tool_outputs:
        try:
            # Add the model's initial response with tool calls
            messages.append(response.message)
            
            # Add each tool output as a separate message
            for tool_output in tool_outputs:
                messages.append({
                    'role': 'tool', 
                    'content': str(tool_output['output']), 
                    'name': tool_output['name']
                })
            
            # Get final response with tool outputs
            logging.info(f"Getting final response from model after tool execution")
            final_response = await client.chat(
                model,
                messages=[{"role": m["role"], "content": m["content"]} 
                        for m in messages if "role" in m and "content" in m]
            )
            
            # Return just the model's content response and the tool info separately
            return final_response.message.content, tool_info
            
        except Exception as e:
            error_message = str(e)
            logging.error(f"Error getting final response after tool execution: {error_message}")
            
            # Create a user-friendly error message
            user_message = "There was an error processing the tool results."
            if "timeout" in error_message.lower():
                user_message = "The model timed out while processing tool results."
            elif "connection" in error_message.lower():
                user_message = "Connection error while getting the final response."
                
            # Return an error message as the final response
            return f"⚠️ {user_message} Please try again.", tool_info
    
    # If no tool calls, return the original response with no tool info
    return response.message.content, None

def format_tool_info_for_display(tool_info: List[str]) -> str:
    """Format tool usage information for display to the user"""
    if not tool_info:
        return ""
        
    return "\n\n💡 Tool Usage Details:" + "".join(tool_info)
