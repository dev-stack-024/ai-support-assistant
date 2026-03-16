"""
ReAct-style agent loop.

Each iteration the LLM produces one of:
  Thought: <reasoning>
  Action: <tool_name>
  Action Input: <input>

or terminates with:
  Final Answer: <reply>

The loop runs until a Final Answer is produced or max_steps is reached.
"""

import re
import logging
from models.message import Message
from services.tools import get_tools_description, run_tool
from config import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

MAX_STEPS = 5  # guard against infinite loops

REACT_SYSTEM_PROMPT = f"""{SYSTEM_PROMPT}

You reason step-by-step before answering. Use the following format strictly:

Thought: <your reasoning about what to do next>
Action: <tool name, one of the available tools>
Action Input: <input to pass to the tool>
Observation: <you will receive the tool result here>
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I now know the final answer.
Final Answer: <your final reply to the user>

If you can answer directly without tools, still show your reasoning:
Thought: <reasoning>
Final Answer: <answer>

{get_tools_description()}
"""


def _parse_llm_output(text: str) -> dict:
    """
    Parse the LLM's raw output into structured fields.

    Returns a dict with keys: thought, action, action_input, final_answer.
    All values default to None if not found.
    """
    result = {
        "thought": None,
        "action": None,
        "action_input": None,
        "final_answer": None,
    }

    thought_match = re.search(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|$)", text, re.DOTALL)
    action_match = re.search(r"Action:\s*(.+?)(?=\n|$)", text)
    input_match = re.search(r"Action Input:\s*(.+?)(?=\nObservation:|\nThought:|$)", text, re.DOTALL)
    final_match = re.search(r"Final Answer:\s*(.+?)$", text, re.DOTALL)

    if thought_match:
        result["thought"] = thought_match.group(1).strip()
    if action_match:
        result["action"] = action_match.group(1).strip()
    if input_match:
        result["action_input"] = input_match.group(1).strip()
    if final_match:
        result["final_answer"] = final_match.group(1).strip()

    return result


def build_agent_messages(history: list[Message], scratchpad: str) -> list[dict]:
    """
    Build the message list for the next LLM call.
    The scratchpad accumulates Thought/Action/Observation turns.
    """
    messages = [{"role": "system", "content": REACT_SYSTEM_PROMPT}]
    messages += [m.to_dict() for m in history]

    if scratchpad:
        # Append the running scratchpad as an assistant turn so the LLM continues from it
        messages.append({"role": "assistant", "content": scratchpad})

    return messages


async def run_agent(history: list[Message], llm_caller) -> tuple[str, str, list[dict]]:
    """
    Run the ReAct agent loop.

    Args:
        history:    Conversation history as Message objects.
        llm_caller: Async callable(messages) -> (reply_text, model_name).

    Returns:
        Tuple of (final_answer, model_name, reasoning_steps).
        reasoning_steps is a list of dicts for observability.
    """
    scratchpad = ""
    reasoning_steps = []
    model_used = ""

    for step in range(MAX_STEPS):
        messages = build_agent_messages(history, scratchpad)
        raw_output, model_used = await llm_caller(messages)

        logger.debug("Agent step %d raw output:\n%s", step + 1, raw_output)

        parsed = _parse_llm_output(raw_output)

        step_record = {
            "step": step + 1,
            "thought": parsed["thought"],
            "action": parsed["action"],
            "action_input": parsed["action_input"],
            "observation": None,
            "final_answer": parsed["final_answer"],
        }

        # If the LLM produced a Final Answer, we're done
        if parsed["final_answer"]:
            reasoning_steps.append(step_record)
            logger.info("Agent finished in %d step(s).", step + 1)
            return parsed["final_answer"], model_used, reasoning_steps

        # If the LLM wants to use a tool, run it and feed the observation back
        if parsed["action"]:
            tool_input = parsed["action_input"] or ""
            observation = run_tool(parsed["action"], tool_input)
            step_record["observation"] = observation

            # Extend the scratchpad with this turn
            scratchpad += (
                f"\nThought: {parsed['thought'] or ''}"
                f"\nAction: {parsed['action']}"
                f"\nAction Input: {tool_input}"
                f"\nObservation: {observation}"
            )

        reasoning_steps.append(step_record)

    # Fallback: max steps reached, ask LLM for a best-effort answer
    logger.warning("Agent reached max steps (%d), requesting final answer.", MAX_STEPS)
    messages = build_agent_messages(history, scratchpad + "\nThought: I must give a final answer now.\nFinal Answer:")
    raw_output, model_used = await llm_caller(messages)
    parsed = _parse_llm_output(raw_output)
    final = parsed["final_answer"] or raw_output.strip()

    reasoning_steps.append({"step": MAX_STEPS + 1, "thought": "max steps reached", "final_answer": final})
    return final, model_used, reasoning_steps
