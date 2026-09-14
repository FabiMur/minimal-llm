"""Tokenize chat conversations into ChatML-formatted training examples."""

from __future__ import annotations

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
