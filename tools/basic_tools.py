"""
Tools available to the ResearchForge agent.

Each tool is a plain Python function. TOOL_SCHEMAS describes them in the
OpenAI/Groq function-calling format so the LLM can decide, per step, whether
and how to call one. TOOL_REGISTRY maps a tool name to a callable that takes
the parsed arguments dict and returns a string result.
"""
import ast
import math
import operator as op
from datetime import datetime

# ---------------------------------------------------------------------------
# Calculator — safe AST-based evaluator (no eval()).
#
# The previous version used eval(expression, {"__builtins__": {}}, allowed)
# which LOOKS sandboxed but isn't: eval() still parses the full Python
# expression grammar, so attribute-chain payloads such as
# "().__class__.__bases__[0].__subclasses__()" can escape the restricted
# builtins. Walking a whitelisted AST instead means only arithmetic nodes we
# explicitly allow can ever execute.
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}
_ALLOWED_UNARYOPS = {ast.USub: op.neg, ast.UAdd: op.pos}
_ALLOWED_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numbers are allowed")

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand))

    if isinstance(node, ast.Call):
        func_name = getattr(node.func, "id", None)
        if func_name in _ALLOWED_FUNCS:
            args = [_safe_eval(a) for a in node.args]
            return _ALLOWED_FUNCS[func_name](*args)
        raise ValueError(f"function '{func_name}' is not allowed")

    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """
    Evaluate a basic arithmetic expression safely.
    Supports + - * / // % ** (or ^ as power) and abs/round/min/max/pow/sqrt.
    """
    try:
        cleaned = expression.replace("^", "**")
        tree = ast.parse(cleaned, mode="eval").body
        result = _safe_eval(tree)
        return str(result)
    except Exception as e:
        return f"Error: could not evaluate '{expression}' ({e})"


def get_current_time() -> str:
    """Returns the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def fake_search(query: str) -> str:
    """Tiny hardcoded fallback used only when real web search is unavailable."""
    query = query.lower().strip()

    if "population" in query and "india" in query:
        return "The current population of India is approximately 1.43 billion (2024 estimate)."
    elif "area" in query and "india" in query:
        return "The total area of India is approximately 3.287 million square kilometers."
    elif "population" in query and ("usa" in query or "united states" in query):
        return "The current population of the United States is approximately 333 million."
    elif "area" in query and ("usa" in query or "united states" in query):
        return "The total area of the United States is approximately 9.8 million square kilometers."
    else:
        return f"No offline fallback data available for: {query}"


def real_search(query: str, max_results: int = 5) -> str:
    """
    Searches the live web via DuckDuckGo (the `ddgs` package).
    Falls back to fake_search if the real search fails or times out.
    """
    try:
        from ddgs import DDGS
        import concurrent.futures

        def do_search():
            results = []
            with DDGS() as ddgs:
                for r in ddgs.text(query, max_results=max_results):
                    results.append(r)
            return results

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(do_search)
            results = future.result(timeout=8)

        if not results:
            print("  -> Real search returned nothing. Using fallback.")
            return fake_search(query)

        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "No title")
            body = r.get("body", r.get("description", "No description"))
            link = r.get("href", r.get("link", ""))
            formatted.append(f"{i}. {title}\n{body}\nSource: {link}\n")

        return "\n".join(formatted)

    except Exception as e:
        print(f"  -> Real search failed ({e}). Using fallback.")
        return fake_search(query)


# ---------------------------------------------------------------------------
# Function-calling schemas + dispatch table.
#
# Passing these to call_llm(..., tools=TOOL_SCHEMAS) lets the model decide,
# per step, whether a tool is needed and with what arguments — replacing the
# old keyword-matching approach (which, e.g., treated any step containing
# the word "present" as a search request).
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live web for current information, facts, or statistics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "A short, specific search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Use for any arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "e.g. '145 * 32 + 90' or 'sqrt(81)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current date and time. Only use when explicitly asked.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

TOOL_REGISTRY = {
    "web_search": lambda args: real_search(args.get("query", "")),
    "calculator": lambda args: calculator(args.get("expression", "")),
    "get_current_time": lambda args: get_current_time(),
}
