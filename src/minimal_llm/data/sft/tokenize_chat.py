"""Tokenize chat conversations into ChatML-formatted training examples."""

from __future__ import annotations

from typing import IO

import numpy as np
from tokenizers import Tokenizer

IGNORE_INDEX = -1


def encode_message(tok: Tokenizer, role: str, content: str, is_target: bool) -> tuple[list[int], list[int]]:
    """Encode one ChatML turn into token ids and aligned loss labels.

    Wraps the turn as "<|im_start|>{role}", a newline, "{content}<|im_end|>", and a trailing
    newline. When `is_target` is True (assistant turns), labels mirror the ids for the content
    and closing delimiter; everywhere else labels are `IGNORE_INDEX`, so the loss only trains
    on assistant output.

    Args:
        tok: Loaded BPE tokenizer with ChatML special tokens.
        role: Message role ("system", "user", or "assistant").
        content: Message text content.
        is_target: Whether this turn's tokens should count toward the loss.

    Returns:
        Tuple of (ids, labels), the same length.
    """
    im_start_id = tok.token_to_id("<|im_start|>")
    im_end_id = tok.token_to_id("<|im_end|>")

    header_ids = [im_start_id, *tok.encode(role + "\n").ids]
    body_ids = tok.encode(content).ids
    footer_ids = [im_end_id, *tok.encode("\n").ids]

    ids = header_ids + body_ids + footer_ids
    if is_target:
        labels = [IGNORE_INDEX] * len(header_ids) + body_ids + footer_ids
    else:
        labels = [IGNORE_INDEX] * len(ids)
    return ids, labels


def encode_conversation(
    tok: Tokenizer, messages: list[dict[str, str]], bos_id: int | None
) -> tuple[list[int], list[int]]:
    """Encode a full conversation into token ids and aligned loss labels.

    Concatenates each message's ChatML turn (see `encode_message`) in order, optionally
    prefixed with a BOS token that is always masked out of the loss.

    Args:
        tok: Loaded BPE tokenizer with ChatML special tokens.
        messages: Ordered role/content messages, e.g. `[{"role": ..., "content": ...}, ...]`.
        bos_id: Optional BOS token id to prepend.

    Returns:
        Tuple of (ids, labels), the same length.
    """
    ids: list[int] = [bos_id] if bos_id is not None else []
    labels: list[int] = [IGNORE_INDEX] if bos_id is not None else []

    for message in messages:
        turn_ids, turn_labels = encode_message(
            tok, message["role"], message["content"], is_target=message["role"] == "assistant"
        )
        ids.extend(turn_ids)
        labels.extend(turn_labels)

    return ids, labels


def write_ids(fh: IO[bytes], ids: list[int], dtype: type[np.uint16] | type[np.int32]) -> int:
    """Write token ids or labels to a binary file.

    Converts a list of ints to a NumPy array with the specified dtype and appends it to
    an open binary file handle.

    Args:
        fh: File handle opened in binary write mode.
        ids: Sequence of token ids or labels.
        dtype: NumPy dtype used to store the values.

    Returns:
        Number of values written.
    """
    if not ids:
        return 0

    arr = np.asarray(ids, dtype=dtype)
    arr.tofile(fh)
    return arr.size
