from pydantic import BaseModel, Field, field_validator
from typing import Literal


class MessageSchema(BaseModel):
    """Schema for a single conversation message."""

    role: Literal["user", "assistant", "system"] = Field(
        ..., description="Role of the message sender"
    )
    content: str = Field(
        ..., min_length=1, max_length=4000, description="Message content"
    )

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, v: str) -> str:
        """Ensure content is not just whitespace."""
        if not v.strip():
            raise ValueError("Message content cannot be blank.")
        return v.strip()


class ChatRequest(BaseModel):
    """Incoming chat request payload."""

    messages: list[MessageSchema] = Field(
        ..., min_length=1, description="Conversation history including the latest user message"
    )
    session_id: str | None = Field(
        default=None, max_length=64, description="Optional session identifier"
    )

    @field_validator("messages")
    @classmethod
    def last_message_must_be_user(cls, v: list[MessageSchema]) -> list[MessageSchema]:
        """Ensure the final message in the list is from the user."""
        if v[-1].role != "user":
            raise ValueError("The last message must have role 'user'.")
        return v


class ReasoningStep(BaseModel):
    """A single step in the agent's reasoning trace."""

    step: int
    thought: str | None = None
    action: str | None = None
    action_input: str | None = None
    observation: str | None = None
    final_answer: str | None = None


class ChatResponse(BaseModel):
    """Response returned to the client."""

    reply: str = Field(..., description="Assistant reply text")
    model: str = Field(..., description="Model used to generate the reply")
    session_id: str | None = Field(default=None, description="Echoed session identifier")
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list, description="Agent reasoning trace")
