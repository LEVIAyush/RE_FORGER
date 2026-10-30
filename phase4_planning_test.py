import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm import call_llm
from tools.basic_tools import calculator, get_current_time, fake_search

# ====================== TOOL SELECTOR ======================
def execute_step(step: str) -> str:
    tool_selector_prompt = """
You are a Tool Selector. You must follow the format EXACTLY.

Available tools:
- fake_search
- calculator
- get_current_time

You will receive one step. Reply with ONLY two lines in this exact format:

Tool: tool_name
Input: short_input

Examples:

Example 1:what 
Step: Identify the most recent population of India
Output:
Tool: fake_search
Input: population of india

Example 2:
Step: Obtain the total land area of India
Output:
Tool: fake_search
Input: area of india

Example 3:
Step: Calculate population density
Output:
Tool: None
Input: None

Example 4:
Step: What time is it?
Output:
Tool: get_current_time
Input: None

Now process the given step. Reply with ONLY the two lines. No explanation.

"""

    messages = [
        {"role": "system", "content": tool_selector_prompt},
        {"role": "user", "content": step}
    ]

    response = call_llm(messages, temperature=0.0)
    decision = response.choices[0].message.content.strip()
    print(f"  LLM Decision → {decision}")

    try:
        lines = decision.split("\n")
        tool_line = lines[0].strip()
        input_line = lines[1].strip() if len(lines) > 1 else "Input: None"

        tool_name = tool_line.replace("Tool:", "").strip().lower()
        tool_input = input_line.replace("Input:", "").strip()

        if tool_name == "fake_search":
            return fake_search(tool_input)
        elif tool_name == "calculator":
            return calculator(tool_input)
        elif tool_name == "get_current_time":
            return get_current_time()
        elif tool_name == "none":
            return "No tool needed"
        else:
            return f"Unknown tool: {tool_name}"
    except Exception as e:
        return f"Error: {e}"


# ====================== MAIN MULTI-STEP AGENT ======================
def run_multi_step_agent(user_question: str):
    print(f"\nUser Question: {user_question}")
    print("=" * 70)

    # ---------- 1. Create Plan ----------
    planning_prompt = """
You are an expert AI Planning Assistant.

Break the user's question into clear, sequential steps.
Do NOT answer the question. Only create a plan.

Output STRICTLY in this format:

Plan:
1. ...
2. ...
3. ...
"""

    messages = [
        {"role": "system", "content": planning_prompt},
        {"role": "user", "content": user_question}
    ]

    plan_response = call_llm(messages, temperature=0.1)
    plan_text = plan_response.choices[0].message.content.strip()
    print("\nGenerated Plan:")
    print(plan_text)
    print("-" * 70)

    # Extract steps
    steps = []
    for line in plan_text.split("\n"):
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith("-")):
            # Remove the number prefix
            step = line.split(".", 1)[-1].strip()
            if step:
                steps.append(step)

    # ---------- 2. Execute each step and store results ----------
    memory = []   # This will store intermediate results

    print("\nExecuting Steps:")
    for i, step in enumerate(steps, 1):
        print(f"\nStep {i}: {step}")
        result = execute_step(step)
        print(f"  Result: {result}")
        memory.append(f"Step {i}: {step}\nResult: {result}")

    # ---------- 3. Generate Final Answer ----------
    print("\n" + "=" * 70)
    print("Generating Final Answer...\n")

    final_prompt = f"""
You are a helpful research assistant.

Here is the original question:
{user_question}

Here is the information collected from the steps:
{chr(10).join(memory)}

Using only the information above, give a clear and final answer to the question.
If calculation is needed, do it carefully.
"""

    final_messages = [
        {"role": "system", "content": "You are a precise research assistant. Give a clear final answer."},
        {"role": "user", "content": final_prompt}
    ]

    final_response = call_llm(final_messages, temperature=0.1)
    final_answer = final_response.choices[0].message.content.strip()

    print("Final Answer:")
    print(final_answer)


# ====================== TEST ======================
if __name__ == "__main__":
    run_multi_step_agent("What is the population density of India?")