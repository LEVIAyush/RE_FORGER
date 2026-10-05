import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.llm import call_llm
from tools.basic_tools import calculator, get_current_time, fake_search

# ====================== STABLE EXECUTE STEP (Rule-based) ======================
def execute_step(step: str) -> str:
    """
    Stable rule-based tool execution.
    """
    step_lower = step.lower()

    # Search related
    if any(word in step_lower for word in ["search", "retrieve", "obtain", "find", "look up", "get the"]):
        if "population" in step_lower and "india" in step_lower:
            return fake_search("population of india")
        elif "area" in step_lower and "india" in step_lower:
            return fake_search("area of india")
        elif "population" in step_lower and ("usa" in step_lower or "united states" in step_lower):
            return fake_search("population of usa")
        elif "area" in step_lower and ("usa" in step_lower or "united states" in step_lower):
            return fake_search("area of usa")
        else:
            return fake_search(step_lower)

    # Calculation related
    elif any(word in step_lower for word in ["calculate", "compute", "divide", "density"]):
        return "Calculation will be done in the final step using collected data."

    # Time related
    elif "time" in step_lower:
        return get_current_time()

    else:
        return "No tool needed for this step."


# ====================== EXTRACT KEY INFO ======================
def extract_key_info(step: str, result: str) -> dict:
    info = {}
    result_lower = result.lower()

    if "population" in step.lower() and "india" in step.lower():
        match = re.search(r"(\d+\.?\d*)\s*billion", result_lower)
        if match:
            info["population_of_india"] = match.group(1) + " billion"

    if "area" in step.lower() and "india" in step.lower():
        match = re.search(r"(\d+\.?\d*)\s*million", result_lower)
        if match:
            info["area_of_india"] = match.group(1) + " million km²"

    return info


# ====================== MAIN AGENT ======================
def run_multi_step_agent(user_question: str):
    print(f"\nUser Question: {user_question}")
    print("=" * 70)

    # 1. Create Plan using LLM
    planning_prompt = """
You are an expert Planning Assistant.
Break the user's question into clear, sequential steps.
Only output the plan in this exact format:

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
        if line and line[0].isdigit():
            step = line.split(".", 1)[-1].strip()
            if step:
                steps.append(step)

    # 2. Execute steps + Structured Memory
    memory = {}
    full_log = []

    print("\nExecuting Steps:")
    for i, step in enumerate(steps, 1):
        print(f"\nStep {i}: {step}")
        result = execute_step(step)
        print(f"  Result: {result}")

        full_log.append(f"Step {i}: {step}\nResult: {result}")

        extracted = extract_key_info(step, result)
        if extracted:
            memory.update(extracted)
            print(f"  → Stored in Memory: {extracted}")

    # Show Memory
    print("\n" + "-" * 70)
    print("Structured Memory:")
    if memory:
        for k, v in memory.items():
            print(f"  {k}: {v}")
    else:
        print("  (empty)")
    print("-" * 70)

    # 3. Final Answer
    print("\nGenerating Final Answer...\n")

    final_prompt = f"""
Original Question: {user_question}

Collected Information:
{chr(10).join(full_log)}

Structured Memory:
{memory}

Using the information above, give a clear and accurate final answer.
If calculation is needed, perform it carefully.
"""

    final_messages = [
        {"role": "system", "content": "You are a precise research assistant. Give a clear final answer."},
        {"role": "user", "content": final_prompt}
    ]

    final_response = call_llm(final_messages, temperature=0.1)
    print("Final Answer:")
    print(final_response.choices[0].message.content.strip())


# ====================== TEST ======================
if __name__ == "__main__":
    run_multi_step_agent("What is the population density of India?")