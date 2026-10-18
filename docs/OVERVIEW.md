# HouWritr — Overview, Architecture & Purpose

## What it is

HouWritr is a **Windows tray app that records meetings and transcribes audio
and video files**, entirely on your own PC. You start a meeting recording or
pick a file from the tray menu; HouWritr turns the speech into a transcript
file, optionally labels who is speaking, and can tidy the text and write
meeting notes with an AI provider you choose.

It started as a fork of [OpenWritr for Windows](https://github.com/trsdn/openwritr-windows),
a push to talk dictation tool. HouWritr removed dictation (hotkeys, pasting at
the cursor, clipboard delivery) and builds on what OpenWritr did well:
fast, private speech recognition on the Snapdragon NPU.

What sets it apart from cloud transcription services:

1. **Recognition runs on your device.** Audio never leaves the machine: no
   cloud, no account, no telemetry.
2. **It uses the NPU.** On Snapdragon X (Copilot+ PC) hardware, Parakeet and
   Whisper run on the Qualcomm Hexagon NPU, far faster and more power
   efficient than the CPU. Intel and AMD machines use the CPU.
3. **Long recordings are first class.** Meetings up to 12 hours are
   transcribed while they are recorded, and files of any length are processed
   in the background with a progress window.

## Purpose / why it exists

Meetings, lectures, interviews and podcasts produce hours of audio that are
hard to search or summarise. Hosted transcription services want the audio on
their servers. HouWritr keeps the audio local, makes use of the NPU that
Snapdragon laptops otherwise leave idle, and produces plain files (`.txt` or
`.md`) that work with any tool, including an AI model of your choice for
notes and reports.

## What it does, concretely

* **Transcribe a file**: WAV, MP3, M4A/AAC, FLAC, OGG and the audio track of
  MP4/M4V/MOV/MKV. The transcript is saved next to the source.
* **Record a meeting**: microphone plus everything the PC plays (WASAPI
  loopback) in one timeline, with pause and resume. Every five minutes the
  finished part is transcribed in the background and appended to an
  in progress file, so a crash loses at most a few minutes and stopping only
  waits for the last part.
* **Speaker labels** (optional): `[m:ss] Speaker N:` blocks from two small
  on-device models (pyannote segmentation and WeSpeaker voice prints), kept
  consistent across the parts of a meeting.
* **Languages**: Parakeet covers 25 European languages; Whisper Large v3 Turbo
  covers about 100, including Chinese and Japanese.
* **AI cleanup** (optional, off by default): the recognised text, never audio,
  is sent in slices to GitHub Copilot, an OpenAI compatible API, or a model on
  this PC (Qwen3 4B on the NPU, or Qwen3 4B or Qwen2.5 7B on the CPU, all
  through Qualcomm GenieX; or a local server you run).
  Hosted providers get three slices at a time. Every slice is checked by an
  integrity validator, and a slice that fails or cannot be sent keeps its
  recognised text and is retried in the background later.
* **Meeting notes** (optional): summary, key points, decisions, action items
  and open questions, saved next to the transcript.
* **Status at a glance**: a tray icon, a small status pill for job progress,
  a meeting pill with the recording clock, and a progress window per file.

## Architecture

HouWritr is a native **Rust** application (no Electron, no web view, no Python
at run time).

```
┌──────────────────────────────────────────────────────────────────┐
│  openwritr.exe (tray process, winit event loop on the main thread) │
│                                                                    │
│  tray menu ── file picker ── decode (Symphonia) → 16 kHz mono      │
│           └── meeting capture: mic (cpal) + loopback (WASAPI)      │
│                     │ rolling 5 minute parts                       │
│                     ▼                                              │
│  ┌──────────────────────────── worker thread ─────────────────┐   │
│  │ speaker diarisation (pyannote + WeSpeaker, CPU, optional)    │   │
│  │        ↓ turns                                               │   │
│  │ ASR in ≤30 s windows cut at quiet points                     │   │
│  │   Parakeet TDT 0.6B v3: encoder on NPU (HTP) or CPU INT8     │   │
│  │   Whisper Large v3 Turbo: encoder + decoder on NPU (QNN)     │   │
│  │        ↓ text                                                │   │
│  │ AI cleanup / notes (optional): Copilot, OpenAI compatible,   │   │
│  │   or local model (GenieX `geniex serve`, NPU or CPU)         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                     │ transcript                                   │
│                     ▼                                              │
│  saved as .txt / .md · background repair of unfinished cleanup     │
│                                                                    │
│  status pill + meeting pill (own Win32 threads, GDI)               │
│  subprocesses: --settings (egui), --transcript-window (egui)       │
└──────────────────────────────────────────────────────────────────┘
```

### The speech engines

| | Parakeet CPU | Parakeet NPU | Whisper NPU |
|---|---|---|---|
| Hardware | any x64 or ARM64 CPU | Snapdragon X Elite | Snapdragon X Elite |
| Model | INT8 ONNX | INT8/INT16 QAIRT context binary (8 s window) | Qualcomm's compiled Large v3 Turbo |
| Languages | 25 | 25 | about 100 |
| Download | ~670 MB | ~650 MB more | ~2.0 GB |

**Parakeet** (NVIDIA Parakeet TDT 0.6B v3) is a 600M parameter Conformer
encoder with a Token and Duration Transducer decoder, with built-in
punctuation and capitalisation. On the NPU the encoder runs as a fixed
8 second window (its dynamic attention mask constant folded, quantised to
INT8 weights / INT16 activations, compiled with Qualcomm AI Hub); longer audio
runs in overlapping chunks whose features are stitched before decoding. The
binary is loaded through ONNX Runtime's QNN execution provider via a thin
direct FFI layer, because the high level Rust bindings crash on this model.

**Whisper** runs its encoder and its key/value cache decoder as precompiled
QNN sessions in 30 second chunks, detecting the language once and reusing it.
The decoder has room for 199 steps; a chunk that runs out (fast speech) or
starts repeating itself is decoded again in two halves, and repetition loops
are cut to one copy.

### Notable design decisions

* **Everything heavy is off the tray thread.** Decoding, diarisation,
  transcription and cleanup run on worker threads; the tray loop only routes
  events, so the menu stays responsive during a two hour job.
* **Windows as subprocesses.** Settings, About and the transcript window are the
  same exe relaunched with a flag, spawned from a background thread
  (`CreateProcessW` on Windows ARM64 with Defender can block for seconds), so a
  slow or crashing window can never stall meeting capture.
* **Partition, never drop.** Speaker turns are turned into windows that cover
  every sample of the recording, cut at the quietest point between turns, so
  labelling can mislabel a word but never lose one.
* **Cleanup never costs text.** Every AI result is validated against the
  recognised text; failures fall back to the recognised text, and unfinished
  sections are retried in the background and written back only if the file is
  untouched.
* **No clipboard, no keystrokes.** Results only ever go to files.
* **Power.** With nothing to do, HouWritr holds no model: the speech engine
  leaves the NPU after 3 idle minutes (never during a meeting) and loads again
  in a few seconds for the next job, and the GenieX server stops after a
  transcript. While working, the app and its GenieX server opt out of Windows
  power throttling, so background jobs keep their speed with no window in
  front.
* **Text size.** The Settings and transcript windows follow Windows' display
  scale and its Accessibility → Text size setting.
* **Observable.** One log per app session and one per transcript, with the
  build commit on the first line and a crash log.

## Privacy

* **No telemetry, no analytics, no accounts.**
* Audio is captured only while a meeting is being recorded (a paused meeting
  releases the microphone), processed locally, and not stored; only the
  transcript is saved.
* Network access is limited to model downloads (Hugging Face, Qualcomm AI Hub
  for the optional local model, the pinned GenieX installer) and, only if you
  turn it on, AI cleanup and notes, which send transcript text to the provider
  you configured with your own credentials.
* Settings, models and logs live under `%LOCALAPPDATA%\OpenWritr\`; API keys
  live in Windows Credential Manager. See [PRIVACY.md](../PRIVACY.md).

## Tech stack

* **Language:** Rust
* **UI:** winit + tray-icon, egui (settings and transcript window), Win32 + GDI
  (pills)
* **Audio:** cpal and WASAPI loopback (capture), Symphonia (file decoding)
* **Inference:** ONNX Runtime (CPU everywhere, QNN on ARM64 for the NPU),
  also used for the speaker diarisation models on the CPU
* **AI cleanup:** GitHub Copilot, OpenAI compatible APIs, Qualcomm GenieX
* **Distribution:** GitHub Releases (installer and portable zip per
  architecture), built by GitHub Actions on Windows ARM64
* **License:** MIT (own code); bundled ONNX Runtime (MIT) and Qualcomm QNN
  runtime (Qualcomm redistributable licence) ship with their notices

## Platform support

* **Windows 11** on **ARM64**: all engines, NPU acceleration on Snapdragon X
  Elite.
* **Windows 11** on **x64**: Parakeet on the CPU.

## Links

* Source: https://github.com/raulhou/HouWritr
* Original project: https://github.com/trsdn/openwritr-windows
* Parakeet NPU model: https://huggingface.co/trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s
* Base model: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
* Whisper NPU model: https://huggingface.co/qualcomm/Whisper-Large-V3-Turbo
