from fastapi import APIRouter, HTTPException
from schemas.chat import ChatRequest, ChatResponse, ReasoningStep
from models.message import Message
from services.llm_service import get_support_reply, get_support_reply_with_steps

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/", response_model=ChatResponse, status_code=200)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint for the AI support assistant.

    Accepts a conversation history and returns the assistant's next reply.
    Pydantic validates the request body automatically before this handler runs.

    Args:
        request: Validated ChatRequest containing message history and optional session_id.

    Returns:
        ChatResponse with the assistant reply, model name, and echoed session_id.

    Raises:
        HTTPException 422: Automatically raised by FastAPI on validation failure.
        HTTPException 502/503/504/500: Propagated from the LLM service layer.
    """
    # Map schema messages to domain model objects
    history = [Message(role=m.role, content=m.content) for m in request.messages]

    # Run the ReAct agent loop
    reply, model_used, steps = await get_support_reply_with_steps(history)

    return ChatResponse(
        reply=reply,
        model=model_used,
        session_id=request.session_id,
        reasoning_steps=[ReasoningStep(**s) for s in steps],
    )


@router.get("/health", status_code=200)
async def health() -> dict:
    """
    Simple health-check endpoint.

    Returns:
        JSON confirming the service is running.
    """
    return {"status": "ok"}
