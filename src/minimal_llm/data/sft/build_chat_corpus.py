"""Build a mixed chat corpus from HuggingFaceH4/no_robots and teknium/OpenHermes-2.5."""

from __future__ import annotations

Message = dict[str, str]
Conversation = list[Message]

_ALLOWED_ROLES = {"system", "user", "assistant"}
_OPEN_HERMES_ROLE_MAP = {"system": "system", "human": "user", "gpt": "assistant"}


def is_valid_conversation(messages: Conversation) -> bool:
    """Check that a conversation has content on every turn and ends with an assistant reply.

    Args:
        messages: Normalized role/content messages.

    Returns:
        True if the conversation is non-empty, every turn has content, the last turn is
        from the assistant, and there is at least one user and one assistant turn.
    """
    if not messages:
        return False
    if any(not m["content"] for m in messages):
        return False
    if messages[-1]["role"] != "assistant":
        return False
    return any(m["role"] == "user" for m in messages) and any(m["role"] == "assistant" for m in messages)
