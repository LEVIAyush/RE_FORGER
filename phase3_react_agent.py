import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm import call_llm
from tools.basic_tools import calculator, get_current_time

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Useful for mathematical calculations only. Use this whenever you need to do any math.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression (example: '145 * 32 + 90')"
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
            "description": "Returns the current date and time. Only use when the user specifically asks for the time or date.",
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

system_prompt = """
You are a careful AI assistant that follows the ReAct pattern.

You have access to tools. Use them only when necessary.

For every response, you must follow this format:

Thought: your reasoning here
Action: tool_name
Action Input: arguments

OR if you can answer without a tool:

Thought: your reasoning here
Final Answer: your final answer

Rules:
- Always think before acting.
- Only use calculator for math questions.
- Only use get_current_time when the user asks for time/date.
- Never call a tool if you already know the answer.
"""

def run_react_agent(user_question: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]

    print(f"\nUser: {user_question}")
    print("=" * 60)

    response = call_llm(messages, tools=tools, temperature=0.1)
    message = response.choices[0].message

    # Case 1: Model called a tool
    if message.tool_calls:
        print("→ Model decided to use a tool")

        messages.append(message)

        for tool_call in message.tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            print(f"Thought + Action: Calling {tool_name}")
            print(f"Action Input: {tool_args}")

            if tool_name == "calculator":
                result = calculator(tool_args.get("expression", ""))
            elif tool_name == "get_current_time":
                result = get_current_time()
            else:
                result = f"Tool {tool_name} not found"

            print(f"Observation: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": tool_name,
                "content": str(result)
            })

        # Get final answer after tool result
        final_response = call_llm(messages, tools=tools, temperature=0.1)
        final_message = final_response.choices[0].message.content
        print(f"\nFinal Answer: {final_message}")

    else:
        # No tool used
        print(message.content)

# ================== TEST ==================
if __name__ == "__main__":
    run_react_agent("What is 145 * 32 + 90?")
    run_react_agent("What is the current time?")
    run_react_agent("Who is the prime minister of India?")
    run_react_agent("What is 25 squared?")