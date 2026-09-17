"""Interactive chat-mode generation for an SFT-finetuned minimal-llm checkpoint."""

import argparse
from pathlib import Path

import torch
from textual import work
from textual.app import App, ComposeResult
from textual.widgets import Input, RichLog
from tokenizers import Tokenizer

from minimal_llm.data.sft.tokenize_chat import encode_message, encode_role_header
from minimal_llm.generate import load_model
from minimal_llm.model import TransformerLM
from minimal_llm.train import get_device


class ChatSession:
    """Maintains ChatML token state for a multi-turn conversation and generates replies.

    Reuses `encode_message`/`encode_role_header` from the SFT tokenization pipeline so the
    prompt fed to the model at inference time is byte-for-byte identical in format to what
    it was fine-tuned on.
    """

    def __init__(
        self,
        model: TransformerLM,
        tok: Tokenizer,
        device: torch.device,
        system: str | None = None,
    ) -> None:
        """Initialize a conversation, optionally opening with a system turn.

        Args:
            model: SFT-finetuned model to generate replies with.
            tok: Loaded BPE tokenizer with ChatML special tokens.
            device: Device to run generation on.
            system: Optional system prompt content.
        """
        self.model = model
        self.tok = tok
        self.device = device
        self.im_end_id = tok.token_to_id("<|im_end|>")
        self.newline_ids = tok.encode("\n").ids

        bos_id = tok.token_to_id("<|bos|>")
        self.ids: list[int] = [bos_id] if bos_id is not None else []
        if system is not None:
            system_ids, _ = encode_message(tok, "system", system, is_target=False)
            self.ids.extend(system_ids)

    def reply(self, user_message: str, num_new_tokens: int, temperature: float, top_k: int | None) -> str:
        """Add a user turn, generate the assistant's reply, and return its text.

        Args:
            user_message: The user's message content.
            num_new_tokens: Max tokens to generate for the reply.
            temperature: Sampling temperature.
            top_k: Only sample from the top k most likely tokens.

        Returns:
            The assistant's reply text (without ChatML delimiters).
        """
        user_ids, _ = encode_message(self.tok, "user", user_message, is_target=False)
        self.ids.extend(user_ids)
        self.ids.extend(encode_role_header(self.tok, "assistant"))

        idx = torch.tensor([self.ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            idx, num_new_tokens, temperature=temperature, top_k=top_k, stop_token_id=self.im_end_id
        )

        new_ids = out[0, idx.shape[1] :].tolist()
        if not new_ids or new_ids[-1] != self.im_end_id:
            # Hit num_new_tokens without producing <|im_end|>; close the turn ourselves so
            # future turns still see well-formed ChatML.
            new_ids.append(self.im_end_id)

        self.ids.extend(new_ids)
        self.ids.extend(self.newline_ids)

        reply_ids = new_ids[:-1] if new_ids[-1] == self.im_end_id else new_ids
        return self.tok.decode(reply_ids)


class ChatApp(App):
    """Textual TUI wrapping a `ChatSession`: a scrollable transcript plus a message input."""

    CSS = """
    RichLog {
        border: none;
        padding: 0 1;
    }
    Input {
        border: none;
        background: transparent;
    }
    """
    BINDINGS = [("ctrl+c", "quit", "Quit")]
    TITLE = "minimal-llm chat"
    THEME = "ansi-dark"

    def __init__(
        self,
        session: ChatSession,
        num_new_tokens: int,
        temperature: float,
        top_k: int | None,
    ) -> None:
        """Store the chat session and generation settings used for every reply.

        Args:
            session: Loaded chat session (model already on device).
            num_new_tokens: Max tokens to generate per reply.
            temperature: Sampling temperature.
            top_k: Only sample from the top k most likely tokens.
        """
        super().__init__()
        self.session = session
        self.num_new_tokens = num_new_tokens
        self.temperature = temperature
        self.top_k = top_k

    def compose(self) -> ComposeResult:
        """Lay out the transcript log and the message input, with no other chrome."""
        yield RichLog(id="transcript", wrap=True, markup=True)
        yield Input(placeholder="Type a message and press Enter…")

    def on_mount(self) -> None:
        """Focus the input box as soon as the app starts."""
        self.query_one(Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Echo the user's message and kick off generation for the reply.

        Args:
            event: Textual event carrying the submitted input value.
        """
        message = event.value.strip()
        if not message:
            return

        input_widget = self.query_one(Input)
        input_widget.value = ""
        input_widget.disabled = True

        self.query_one(RichLog).write(f"[bold]you>[/] {message}")
        self.generate_reply(message)

    @work(exclusive=True, thread=True)
    def generate_reply(self, message: str) -> None:
        """Run the (blocking) model generation off the UI thread.

        Args:
            message: The user's message to reply to.
        """
        reply = self.session.reply(message, self.num_new_tokens, self.temperature, self.top_k)
        self.call_from_thread(self._show_reply, reply)

    def _show_reply(self, reply: str) -> None:
        """Write the assistant's reply to the transcript and re-enable input.

        Args:
            reply: Decoded assistant reply text.
        """
        self.query_one(RichLog).write(f"[bold]assistant>[/] {reply}\n")
        input_widget = self.query_one(Input)
        input_widget.disabled = False
        input_widget.focus()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for chat-mode generation."""
    parser = argparse.ArgumentParser(description="Chat with an SFT-finetuned minimal-llm checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to an SFT checkpoint.")
    parser.add_argument("--tokenizer", type=Path, default=Path("artifacts/tokenizer.json"))
    parser.add_argument("--system", type=str, default=None, help="Optional system prompt for the conversation.")
    parser.add_argument("--num_new_tokens", type=int, default=256, help="Max tokens to generate per reply.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature.")
    parser.add_argument("--top_k", type=int, default=50, help="Only sample from the top k most likely tokens.")
    return parser.parse_args()


def main() -> None:
    """Load an SFT checkpoint and start the interactive chat TUI."""
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    model = load_model(args.checkpoint, device)
    print(f"Parameters: {model.count_parameters() / 1e6:.1f}M")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    session = ChatSession(model, tokenizer, device, system=args.system)

    app = ChatApp(session, args.num_new_tokens, args.temperature, args.top_k)
    app.run()


if __name__ == "__main__":
    main()
