from ollama import ChatResponse
from typing import Dict, Any, List

# Import the functions and dictionary from tool_functions.py
from tool_functions import available_functions

async def process_tool_calls(client, model: str, messages: List[Dict[str, Any]], 
                            response: ChatResponse) -> str:
    """
    Process tool calls from the model response and get the final response.
    
    Args:
        client: The Ollama client
        model: The model name
        messages: The message history
        response: The model's response containing tool calls
        
    Returns:
        str: The final response after processing tool calls
    """
    tool_outputs = []
    tool_info = []
    
    # Process each tool call
    if response.message.tool_calls:
        for tool in response.message.tool_calls:
            if function_to_call := available_functions.get(tool.function.name):
                # Create tool call info string
                tool_call_info = (
                    f"\n🔧 Tool Called: {tool.function.name}\n"
                    f"📝 Parameters: {tool.function.arguments}\n"
                )
                
                # Call the function with the provided arguments
                tool_output = function_to_call(**tool.function.arguments)
                
                # Store the output for adding to messages
                tool_outputs.append({
                    'output': tool_output,
                    'name': tool.function.name
                })
                
                # Add result to tool info for display
                tool_call_info += f"📊 Result: {tool_output}\n"
                tool_info.append(tool_call_info)
    
    # If we have tool outputs, get final response from model
    if tool_outputs:
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
        final_response = await client.chat(
            model,
            messages=[{"role": m["role"], "content": m["content"]} 
                    for m in messages if "role" in m and "content" in m]
        )
        
        # Format the complete response
        complete_response = final_response.message.content
        if tool_info:
            complete_response += "\n\n💡 Tool Usage Details:" + "".join(tool_info)
        
        return complete_response
    
    # If no tool calls, return the original response
    return response.message.content
