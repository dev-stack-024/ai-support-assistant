import httpx
from fastapi import HTTPException

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    MODEL_NAME,
    TEMPERATURE,
    MAX_TOKENS,
)
from models.message import Message
from services.agent import run_agent


async def call_openrouter(messages: list[dict]) -> tuple[str, str]:
    """
    Send a chat completion request to OpenRouter.

    Args:
        messages: Serialized message list (system + history + scratchpad).

    Returns:
        Tuple of (reply_text, model_name).

    Raises:
        HTTPException 502/503/504/500 on failure.
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="OPENROUTER_API_KEY is not configured on the server.",
        )

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OPENROUTER_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
            )
    except httpx.ConnectError:
        raise HTTPException(
            status_code=503,
            detail="Unable to reach OpenRouter API. Check your network connection.",
        )
    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="OpenRouter API request timed out.",
        )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"OpenRouter error {response.status_code}: {response.text}",
        )

    data = response.json()

    try:
        reply = data["choices"][0]["message"]["content"]
        model_used = data.get("model", MODEL_NAME)
    except (KeyError, IndexError) as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected response structure from OpenRouter: {exc}",
        )

    return reply, model_used


async def get_support_reply(history: list[Message]) -> tuple[str, str]:
    """
    Run the ReAct agent loop and return the final answer.

    Args:
        history: Conversation history as domain Message objects.

    Returns:
        Tuple of (final_answer, model_name).
    """
    final_answer, model_used, _ = await run_agent(history, call_openrouter)
    return final_answer, model_used


async def get_support_reply_with_steps(history: list[Message]) -> tuple[str, str, list[dict]]:
    """
    Run the ReAct agent loop and return the final answer plus reasoning steps.

    Returns:
        Tuple of (final_answer, model_name, reasoning_steps).
    """
    return await run_agent(history, call_openrouter)
