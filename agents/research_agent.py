import json
import os
import re
from datetime import datetime, timezone

from memory.store import MemoryStore
from tools.basic_tools import TOOL_REGISTRY, TOOL_SCHEMAS
from utils.llm import call_llm


class ResearchAgent:
    def __init__(self):
        # Short-term / working memory: reset every run() call, by design.
        self.full_log = []
        self.conversation_history = []

        # Long-term / persistent memory: backed by a JSON file on disk, so
        # it survives across questions AND across app restarts. This fixes
        # the previous behaviour where `self.memory = {}` was wiped at the
        # start of every single run().
        self.memory_store = MemoryStore()

        # For UI process display
        self.last_plan = []
        self.last_steps_results = []
        self.last_recalled_memory = []
        self.last_draft = ""
        self.last_reflection = ""
        self.last_final_answer = ""
        self.last_report_path = None
        self.last_report_content = ""

    # -------------------------------------------------------------------
    # 1. Planning
    # -------------------------------------------------------------------
    def create_plan(self, question: str) -> list:
        """Forces the agent to create a proper research plan."""
        planning_prompt = """
You are a careful Research Planning Assistant.

Your job is to break the user's question into clear research steps.
You must ALWAYS create a plan, even if you think you already know the answer.
Do NOT give the final answer in the plan.
Do NOT skip research steps.

Rules:
- Always include steps to search for up-to-date information.
- Keep steps specific and actionable.
- Output ONLY in this format:

Plan:
1. ...
2. ...
3. ...
"""
        messages = [{"role": "system", "content": planning_prompt}]
        messages.extend(self.conversation_history[-4:])
        messages.append({"role": "user", "content": question})

        try:
            response = call_llm(messages, temperature=0.1)
            plan_text = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  -> Planning failed, using a default plan. ({e})")
            plan_text = ""

        print("\nGenerated Plan:")
        print(plan_text)
        print("-" * 70)

        steps = self._parse_plan(plan_text)

        if not steps:
            steps = [
                f"Search for the latest information about: {question}",
                "Extract the most relevant and recent data",
                "Present the final answer with sources",
            ]

        return steps

    @staticmethod
    def _parse_plan(plan_text: str) -> list:
        """
        Extracts step strings from the model's plan text.

        Handles "1. step", "1) step", and "- step" formats. The previous
        version only handled "1. step" via `line.split(".", 1)` — any other
        numbering style (e.g. "1) step") left the "1) " prefix stuck to the
        step text because split(".", 1) found no "." to split on.
        """
        steps = []
        for line in plan_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            if line[0].isdigit():
                match = re.match(r"^\d+[\.\)]\s*(.*)", line)
                step = match.group(1).strip() if match else line
            elif line.startswith("-") or line.startswith("*"):
                step = line[1:].strip()
            else:
                continue

            if step:
                steps.append(step)
        return steps

    # -------------------------------------------------------------------
    # 2. Execution — real LLM-driven tool calling (replaces keyword matching)
    # -------------------------------------------------------------------
    def execute_step(self, step: str) -> dict:
        """
        Executes one research step. The LLM itself decides, per step,
        whether a tool is needed and which one — using real function
        calling — rather than matching keywords like "search"/"present"/
        "data" against the step text (which previously misfired, e.g. any
        step containing the word "present" triggered a web search, and
        calculation steps were never actually computed).

        Returns {"result": str, "tool_calls": [{"tool", "input", "output"}]}
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are executing ONE step of a research plan. "
                    "Call a tool if it would genuinely help: web_search for "
                    "current facts/statistics, calculator for arithmetic, "
                    "get_current_time for date/time questions. "
                    "If no tool is needed, answer the step directly and "
                    "concisely (1-3 sentences) using your own knowledge."
                ),
            },
            {"role": "user", "content": step},
        ]

        try:
            response = call_llm(messages, tools=TOOL_SCHEMAS, temperature=0.1)
        except Exception as e:
            return {"result": f"(step failed: {e})", "tool_calls": []}

        message = response.choices[0].message
        tool_trace = []

        if message.tool_calls:
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}

                handler = TOOL_REGISTRY.get(name)
                try:
                    output = handler(args) if handler else f"Unknown tool '{name}'"
                except Exception as e:
                    output = f"Tool '{name}' raised an error: {e}"

                print(f"  -> {name}({args}) = {str(output)[:200]}")
                tool_trace.append({"tool": name, "input": args, "output": output})

            combined_result = "\n".join(f"[{t['tool']}] {t['output']}" for t in tool_trace)
            return {"result": combined_result, "tool_calls": tool_trace}

        content = (message.content or "").strip()
        return {"result": content or "No tool needed; no additional info.", "tool_calls": []}

    # -------------------------------------------------------------------
    # 3. Synthesis
    # -------------------------------------------------------------------
    def generate_final_answer(self, question: str) -> str:
        """Generates the final answer using this run's findings, relevant
        prior research recalled from persistent memory, and conversation
        history."""
        recalled = self.memory_store.recall(question, k=5)
        self.last_recalled_memory = recalled
        recalled_text = (
            "\n".join(f"- {r['step']}: {r['result']}" for r in recalled)
            if recalled
            else "(none found)"
        )

        final_prompt = f"""
Original Question: {question}

Collected information from this research session:
{chr(10).join(self.full_log)}

Potentially relevant prior research from earlier sessions (may or may not apply):
{recalled_text}

Using the conversation history and the information above, give a clear and accurate final answer.
If the user is referring to previous context (like "choose one" or "those countries"), use that context.
"""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise research assistant. Use conversation "
                    "history to understand context. Give a clear final answer."
                ),
            }
        ]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": final_prompt})

        try:
            response = call_llm(messages, temperature=0.1)
            return response.choices[0].message.content.strip()
        except Exception as e:
            fallback = "\n\n".join(self.full_log) or "No information was gathered."
            return (
                f"(Could not synthesize a final answer due to an error: {e})\n\n"
                f"Raw findings collected:\n{fallback}"
            )

    def reflect(self, question: str, draft_answer: str) -> str:
        """Critiques and improves the draft answer. Falls back to the draft
        itself if the reflection call fails, instead of losing the answer."""
        prompt = f"""
Improve the following research answer.

Question: {question}

Draft Answer:
{draft_answer}

Make the answer clearer, more accurate, and better structured.
Add sources if possible.
Return ONLY the improved answer.
"""
        messages = [
            {
                "role": "system",
                "content": "You improve research answers. Return only the improved answer.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            response = call_llm(messages, temperature=0.2)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"  -> Reflection failed, keeping draft answer. ({e})")
            return draft_answer

    # -------------------------------------------------------------------
    # 4. Reporting
    # -------------------------------------------------------------------
    def _extract_sources(self) -> list:
        """Pulls unique source URLs out of this run's web_search results so
        they can be listed in the saved report (previously discarded)."""
        sources = []
        for item in self.last_steps_results:
            for tc in item.get("tool_calls", []):
                if tc["tool"] != "web_search":
                    continue
                for line in str(tc["output"]).split("\n"):
                    line = line.strip()
                    if line.startswith("Source:"):
                        url = line.split("Source:", 1)[1].strip()
                        if url and url not in sources:
                            sources.append(url)
        return sources

    def save_report(self, question: str, answer: str) -> str:
        """Saves the research report as a Markdown file and returns its path."""
        os.makedirs("data", exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        safe_question = "".join(c if c.isalnum() else "_" for c in question)[:50]
        filename = f"data/report_{safe_question}_{timestamp}.md"
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        sources = self._extract_sources()

        content = f"""# ResearchForge Report

**Question:** {question}

**Date:** {current_time}

---

## Final Answer

{answer}

---

## Sources

"""
        if sources:
            for url in sources:
                content += f"- {url}\n"
        else:
            content += "- No web sources were used for this answer.\n"

        content += "\n---\n*Generated by ResearchForge*\n"

        self.last_report_content = content

        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"\nReport saved to: {filename}")
        return filename

    # -------------------------------------------------------------------
    # 5. Orchestration
    # -------------------------------------------------------------------
    def run(self, question: str, on_event=None):
        """
        Main method to run the full agent.

        `on_event(event_name, payload)` is an optional callback fired after
        each stage completes (plan / step / draft / reflection / final /
        done), so a UI can render progress live instead of waiting for the
        entire pipeline to finish. It is never allowed to crash the agent:
        any exception raised inside the callback is swallowed.
        """

        def emit(event, payload=None):
            if on_event:
                try:
                    on_event(event, payload or {})
                except Exception as e:
                    print(f"  -> on_event callback raised (ignored): {e}")

        print(f"\nUser Question: {question}")
        print("=" * 70)

        # Reset only this run's working memory — conversation_history and
        # the persistent memory_store are deliberately NOT reset here.
        self.full_log = []
        self.last_plan = []
        self.last_steps_results = []
        self.last_recalled_memory = []
        self.last_draft = ""
        self.last_reflection = ""
        self.last_final_answer = ""
        self.last_report_path = None

        # 1. Create Plan
        steps = self.create_plan(question)
        self.last_plan = steps
        emit("plan", {"steps": steps})

        # 2. Execute Steps
        print("\nExecuting Steps:")
        for i, step in enumerate(steps, 1):
            print(f"\nStep {i}: {step}")
            exec_result = self.execute_step(step)
            result_text = exec_result["result"]
            print(f"  Result: {result_text}")

            self.full_log.append(f"Step {i}: {step}\nResult: {result_text}")
            self.memory_store.add(topic=question, step=step, result=result_text)

            step_record = {
                "step": step,
                "result": result_text,
                "tool_calls": exec_result["tool_calls"],
            }
            self.last_steps_results.append(step_record)
            emit("step", {"index": i, "total": len(steps), **step_record})

        # 3. Generate Draft Answer (also recalls relevant persistent memory)
        print("\nGenerating Draft Answer...\n")
        draft_answer = self.generate_final_answer(question)
        self.last_draft = draft_answer
        print("Draft Answer:")
        print(draft_answer)
        emit("draft", {"text": draft_answer, "recalled": self.last_recalled_memory})

        # 4. Reflect / Critique
        print("\nReflecting on the draft answer...\n")
        reflection = self.reflect(question, draft_answer)
        self.last_reflection = reflection
        print("Reflection:")
        print(reflection)
        emit("reflection", {"text": reflection})

        final_answer = reflection.strip() if reflection.strip() else draft_answer
        self.last_final_answer = final_answer
        print("\nFinal Improved Answer:")
        print(final_answer)
        emit("final", {"text": final_answer})

        # 5. Update Conversation History
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": final_answer})
        if len(self.conversation_history) > 12:
            self.conversation_history = self.conversation_history[-12:]

        # 6. Save Report (never let a disk error lose the answer already produced)
        try:
            self.last_report_path = self.save_report(question, final_answer)
        except Exception as e:
            print(f"  -> Could not save report: {e}")
            self.last_report_path = None

        emit("done", {"path": self.last_report_path})

        return final_answer
