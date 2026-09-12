"""Build a mixed chat corpus from HuggingFaceH4/no_robots and teknium/OpenHermes-2.5."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from datasets import IterableDataset

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


def yield_no_robots(ds: IterableDataset) -> Iterator[Conversation]:
    """Yield cleaned conversations from a streamed HuggingFaceH4/no_robots dataset.

    Extracts the "messages" field from each record, strips whitespace, and yields
    conversations that pass `is_valid_conversation`.

    Args:
        ds: Iterable of records where record["messages"] is a list of role/content dicts.

    Yields:
        Normalized conversations.
    """
    for record in ds:
        messages = record.get("messages")
        if not isinstance(messages, list):
            continue

        conv: Conversation = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role not in _ALLOWED_ROLES or not isinstance(content, str):
                conv = []
                break
            conv.append({"role": role, "content": content.strip()})

        if is_valid_conversation(conv):
            yield conv


def yield_open_hermes(ds: IterableDataset) -> Iterator[Conversation]:
    """Yield cleaned conversations from a streamed teknium/OpenHermes-2.5 dataset.

    Extracts the "conversations" field (ShareGPT-style `from`/`value` turns) from each
    record, maps `from` to a role, strips whitespace, and yields conversations that pass
    `is_valid_conversation`.

    Args:
        ds: Iterable of records where record["conversations"] is a list of from/value dicts.

    Yields:
        Normalized conversations.
    """
    for record in ds:
        turns = record.get("conversations")
        if not isinstance(turns, list):
            continue

        conv: Conversation = []
        for turn in turns:
            role = _OPEN_HERMES_ROLE_MAP.get(turn.get("from"))
            content = turn.get("value")
            if role is None or not isinstance(content, str):
                conv = []
                break
            conv.append({"role": role, "content": content.strip()})

        if is_valid_conversation(conv):
            yield conv


def take_n(elements: Iterable[Conversation], n: int) -> Iterator[Conversation]:
    """Take at most the first n elements from an iterable of conversations.

    Args:
        elements: An iterable producing conversations.
        n: Maximum number of elements to take.

    Yields:
        Up to n elements from the input iterable.
    """
    for i, x in enumerate(elements):
        if i >= n:
            break
        yield x
