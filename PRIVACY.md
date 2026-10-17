# HouWritr Privacy Policy

*Last updated: October 2026*

HouWritr is a local-first tool for recording meetings and transcribing audio
and video files. Audio capture and speech recognition run **entirely on your
device**.

## What we collect

**Nothing.** HouWritr has no telemetry, no analytics, no accounts, and no
servers operated by us.

## How your audio is processed

- Audio is captured from your microphone and the PC's sound output **only
  while you record a meeting** (started and stopped from the tray menu; a
  paused meeting releases the microphone). Audio and video files are read only
  when you choose them.
- Transcription runs **locally on your device** (NVIDIA Parakeet on your CPU
  or NPU, or Whisper on the NPU). Your audio never leaves your machine.
- HouWritr never pastes text or puts it on the Windows clipboard.
- **Files you transcribe and meetings you record are saved as transcript
  files**: next to the imported file, and meetings under
  `Documents\OpenWritr\Meetings`. A transcript that cannot be saved there
  goes to `Documents\HouWritr`. They stay on your PC until you delete them.

## Network access

HouWritr connects to the internet only for:

1. **One-time model download** from Hugging Face (`huggingface.co`) on first
   launch — this fetches the speech-recognition model files. No personal
   data is sent; this is a plain file download.
2. **Optional speaker labels** — *off by default*. If you turn on *Label
   speakers in file and meeting transcripts*, two small models (about 32 MB)
   are downloaded once from Hugging Face. The labelling itself runs entirely
   on your device; no audio or text is sent anywhere for it.
3. **Optional text cleanup ("Enhance")** — *off by default*. If turned on,
   the **recognized text** (never audio) is sent to GitHub Copilot or your
   configured OpenAI-compatible API for grammar and punctuation cleanup using
   your own credentials. The provider's privacy policy applies. Leave
   enhancement **Off** to keep all processing local.
   For GitHub Copilot, HouWritr uses the GitHub sign-in already on your PC: a
   `COPILOT_GITHUB_TOKEN`, `GH_TOKEN` or `GITHUB_TOKEN` variable, the token
   GitHub Copilot CLI saved (`copilot login`), or the GitHub CLI's. It is sent
   only to GitHub. An API key for another service is sent only to the address
   you entered, and never over unencrypted `http` to another machine.
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
`%LOCALAPPDATA%\OpenWritr\` on your machine (the folder keeps the name of
the project HouWritr started from, so existing installs keep their models).
They never leave it unless you choose to share a diagnostics export, which
removes API keys and prompt text first. API keys you enter are kept in Windows
Credential Manager, not in the settings file. Deleting that folder (or
uninstalling) removes the rest.

## Contact

Questions: open an issue at
https://github.com/raulhou/HouWritr/issues
