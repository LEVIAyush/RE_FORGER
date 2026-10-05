import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm import call_llm
from tools.basic_tools import calculator, get_current_time

# Tool definitions (This is how the model understands tools)
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Useful for mathematical calculations. Use this for any math problem.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate (example: '145 * 32 + 90')"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current date and time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]

available_tools = {
    "calculator": calculator,
    "get_current_time": get_current_time
}

def run_agent(user_question: str):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use tools when needed. For any calculation, always use the calculator tool."
        },
        {
            "role": "user",
            "content": user_question
        }
    ]

    print(f"\nUser: {user_question}")
    print("-" * 60)

    # First call
    response = call_llm(messages, tools=tools, temperature=0.1)
    response_message = response.choices[0].message

    # Check if model wants to call a tool
    if response_message.tool_calls:
        print("→ Model decided to use a tool")

        # Add the assistant's tool call message
        messages.append(response_message)

        for tool_call in response_message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"→ Calling tool: {tool_name}")
            print(f"→ Arguments: {tool_args}")

            # Execute the tool
            if tool_name == "calculator":
                tool_result = calculator(tool_args.get("expression", ""))
            elif tool_name == "get_current_time":
                tool_result = get_current_time()
            else:
                tool_result = f"Tool {tool_name} not found"

            print(f"→ Tool Result: {tool_result}")

            # Add tool result back to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": str(tool_result)
            })

        # Call the model again with tool results
        final_response = call_llm(messages, tools=tools, temperature=0.1)
        final_answer = final_response.choices[0].message.content
        print(f"\nFinal Answer: {final_answer}")

    else:
        # No tool was needed
        print(f"Assistant: {response_message.content}")

# ==================== TEST ====================
if __name__ == "__main__":
    run_agent("What is 145 * 32 + 90?")
    run_agent("What is the current time?")
    run_agent("Who is the prime minister of India?")
    run_agent("Calculate 89 multiplied by 12")