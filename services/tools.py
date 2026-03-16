"""
Tool registry for the ReAct agent.
Each tool is a callable that accepts a string input and returns a string output.
"""

import math
from datetime import datetime, timezone


def tool_calculator(expression: str) -> str:
    """
    Safely evaluate a basic math expression.
    Example input: "2 + 2 * 10"
    """
    try:
        # Restrict to safe math operations only
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed["__builtins__"] = {}
        result = eval(expression, allowed)  # noqa: S307
        return str(result)
    except Exception as exc:
        return f"Calculator error: {exc}"


def tool_datetime(_: str) -> str:
    """Return the current UTC date and time."""
    now = datetime.now(timezone.utc)
    return now.strftime("Current UTC date/time: %Y-%m-%d %H:%M:%S")


def tool_knowledge_base(query: str) -> str:
    """
    Simple static knowledge base lookup.
    Extend this with a real vector DB or FAQ store as needed.
    """
    kb = {
        "refund": "Refunds are processed within 5–7 business days to the original payment method.",
        "shipping": "Standard shipping takes 3–5 business days. Express shipping is 1–2 business days.",
        "contact": "You can reach support at support@example.com or call 1-800-000-0000.",
        "password": "To reset your password, click 'Forgot Password' on the login page.",
        "cancel": "You can cancel your subscription at any time from Account Settings > Subscription.",
    }
    query_lower = query.lower()
    for key, answer in kb.items():
        if key in query_lower:
            return answer
    return "No relevant information found in the knowledge base."


# Registry maps tool name → (callable, description)
TOOLS: dict[str, tuple[callable, str]] = {
    "calculator": (
        tool_calculator,
        "Evaluate a math expression. Input: a math expression string e.g. '12 * 8 + 4'.",
    ),
    "datetime": (
        tool_datetime,
        "Get the current UTC date and time. Input: empty string.",
    ),
    "knowledge_base": (
        tool_knowledge_base,
        "Look up support information. Input: a keyword or question about refunds, shipping, contact, password, or cancellation.",
    ),
}


def get_tools_description() -> str:
    """Return a formatted string describing all available tools for the system prompt."""
    lines = ["Available tools:"]
    for name, (_, desc) in TOOLS.items():
        lines.append(f"  - {name}: {desc}")
    return "\n".join(lines)


def run_tool(name: str, input_str: str) -> str:
    """Execute a tool by name and return its output."""
    if name not in TOOLS:
        return f"Unknown tool '{name}'. Available tools: {', '.join(TOOLS.keys())}"
    fn, _ = TOOLS[name]
    return fn(input_str)
