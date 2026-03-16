from dataclasses import dataclass


@dataclass
class Message:
    """
    Internal domain model representing a single chat message.
    Decoupled from the Pydantic schema so the service layer
    stays independent of HTTP concerns.
    """

    role: str   # "user" | "assistant" | "system"
    content: str

    def to_dict(self) -> dict:
        """Serialize to the dict format expected by the OpenRouter API."""
        return {"role": self.role, "content": self.content}
