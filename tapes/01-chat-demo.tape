# minimal-llm — chat TUI demo
#
#   vhs tapes/01-chat-demo.tape        (run from the repo root)
#
# Shows the Textual chat REPL against the SFT-finetuned checkpoint: multi-turn
# ChatML conversation, generation running off the UI thread (input disables
# while replying, re-enables when done).

Output tapes/01-chat-demo.gif

Require uv

Set Shell "bash"
Set FontSize 16
Set Width 1200
Set Height 720
Set Padding 20
Set TypingSpeed 40ms
Set Theme "Catppuccin Mocha"

# Warm everything (venv resolution, torch import, and the OS page cache for
# the 4.7GB checkpoint) by running the exact same command once, hidden, then
# quitting it — otherwise the recorded run opens on a long dead "Device: mps"
# screen while the checkpoint cold-reads off disk.
Hide
Type "uv run python -m minimal_llm.chat --checkpoint artifacts/checkpoints/sft-run1/best.pt --tokenizer artifacts/tokenizer.json --num_new_tokens 80"
Enter
Wait+Screen@60s /Type a message/
Ctrl+C
Sleep 1s
Type "clear"
Enter
Sleep 1s
Show

Type "uv run python -m minimal_llm.chat --checkpoint artifacts/checkpoints/sft-run1/best.pt --tokenizer artifacts/tokenizer.json --num_new_tokens 80"
Enter
Wait+Screen@30s /Type a message/
Sleep 500ms

Type "What is the capital of France?"
Sleep 500ms
Enter
Sleep 6s

Type "Write a short poem about the ocean."
Sleep 500ms
Enter
Sleep 6s

Sleep 2s
Ctrl+C
Sleep 1s
