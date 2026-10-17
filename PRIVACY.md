# OpenWritr Privacy Policy

*Last updated: July 2026*

OpenWritr is a local-first push-to-talk voice-to-text tool. Audio capture and
speech recognition run **entirely on your device**.

## What we collect

**Nothing.** OpenWritr has no telemetry, no analytics, no accounts, and no
servers operated by us.

## How your voice is processed

- Audio is captured from your microphone **only while you hold the hotkey**.
- Transcription runs **locally on your device** (NVIDIA Parakeet model on
  your CPU or NPU). Your audio never leaves your machine.
- The recognized text is pasted at your cursor or copied to the Windows
  clipboard, depending on your Settings choice. OpenWritr does not keep a
  transcript history.

## Network access

OpenWritr connects to the internet only for:

1. **One-time model download** from Hugging Face (`huggingface.co`) on first
   launch — this fetches the speech-recognition model files. No personal
   data is sent; this is a plain file download.
2. **Optional speaker labels** — *off by default*. If you turn on *Label
   speakers in file and meeting transcripts*, two small models (about 32 MB)
   are downloaded once from Hugging Face. The labelling itself runs entirely
   on your device; no audio or text is sent anywhere for it.
3. **Optional text cleanup ("Enhance")** — *off by default*. You can run it
   only when Shift is additionally held or for every recording. If enabled,
   the **recognized text** (never audio) is sent to GitHub Copilot or your
   configured OpenAI-compatible API for grammar and punctuation cleanup using
   your own credentials. The provider's privacy policy applies. Set enhancement
   to **Never** to keep all processing local.
4. **Optional local model for cleanup** — choose *Local model (on this PC)*
   in Settings and the text is processed by a model running on your own
   computer (on the Snapdragon NPU, or a server such as LM Studio on
   `localhost`); the text is never sent over the internet. Setting it up
   downloads, only when you press the button, the official Qualcomm GenieX
   installer from GitHub (about 62 MB, checked against a pinned SHA-256) and
   the model from Qualcomm AI Hub (several GB). Only loopback addresses are
   accepted for a local model.

## Data stored on your device

Settings, downloaded models, and a local diagnostic log are stored under
`%LOCALAPPDATA%\OpenWritr\` on your machine. They never leave it. Deleting
that folder (or uninstalling) removes them.

## Contact

Questions: open an issue at
https://github.com/trsdn/openwritr-windows/issues
