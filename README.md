# HouWritr

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Windows 11](https://img.shields.io/badge/Windows-11_24H2-0078D4?logo=windows)](https://github.com/raulhou/HouWritr)
[![Rust](https://img.shields.io/badge/Rust-1.85-DEA584?logo=rust&logoColor=white)](https://www.rust-lang.org)
[![Snapdragon X Elite](https://img.shields.io/badge/Snapdragon-X_Elite-3253DC?logo=qualcomm&logoColor=white)](https://www.qualcomm.com/products/mobile/snapdragon/pcs-and-tablets/snapdragon-x-elite)
[![Hexagon NPU](https://img.shields.io/badge/Hexagon-NPU-FF2A00?logo=qualcomm&logoColor=white)](https://huggingface.co/trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s)
[![Release](https://img.shields.io/github/v/release/raulhou/HouWritr)](https://github.com/raulhou/HouWritr/releases/latest)

Meeting recording and audio/video transcription for Windows, on your own PC.
Speech recognition runs locally with **NVIDIA Parakeet TDT 0.6B v3** (CPU or
Qualcomm Hexagon NPU) or **Whisper Large v3 Turbo** (Snapdragon X Elite NPU).
Speaker labels run on the device too. Optional AI cleanup and meeting notes
use **GitHub Copilot**, any OpenAI compatible API, or a **model on this PC**
(the Snapdragon NPU through Qualcomm GenieX, or a local server such as LM
Studio).

**[Releases](https://github.com/raulhou/HouWritr/releases)** · **[Overview](docs/OVERVIEW.md)** · **[NPU model on HF](https://huggingface.co/trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s)**

HouWritr started as a fork of [OpenWritr for Windows](https://github.com/trsdn/openwritr-windows)
by trsdn (MIT licence), a push to talk dictation tool, and is now developed
separately. HouWritr dropped dictation and focuses on two jobs: **recording
meetings** and **transcribing audio and video files**, each saved as a
transcript file.

## Contents

* [Quick start](#quick-start)
* [Using HouWritr](#using-houwritr)
* [Transcribing a file](#transcribing-a-file)
* [Recording a meeting](#recording-a-meeting)
* [Speaker labels](#speaker-labels)
* [AI cleanup and meeting notes](#ai-cleanup-and-meeting-notes)
* [Settings](#settings)
* [Files and folders](#files-and-folders)
* [Logs and diagnostics](#logs-and-diagnostics)
* [Architecture](#architecture)
* [NPU pipeline](#npu-pipeline), [Models](#models), [Licenses](#licenses), [Development](#development), [Releases](#releases)

## Quick start

HouWritr is not in the Microsoft Store (the OpenWritr listing there is the
original project's). Pick the build for your CPU; both install per user, with
no admin rights needed:

| Your machine | Build | Engines |
|---|---|---|
| **Snapdragon X Elite** (Surface Pro 11, etc.) | `…arm64…` | Parakeet CPU and NPU, Whisper NPU |
| **Intel / AMD** laptop | `…x64…` | Parakeet CPU |

Not sure? Snapdragon laptops report "ARM based processor" in Settings →
System → About. Everything else is x64.

**Installer.** Download `houwritr-windows-<arch>-vX.Y.Z-setup.exe` from
[Releases](https://github.com/raulhou/HouWritr/releases/latest) and run it.
It adds a Start menu entry, an optional start at sign in entry and an
uninstaller under Settings → Apps.

**Portable zip.** Download `houwritr-windows-<arch>-vX.Y.Z.zip`, unzip it
into any folder and run `houwritr.exe`. The app finds its DLLs next to the
exe. Start at sign in is available here too (Settings → Startup) and is
repaired automatically if you move the folder.

> **Windows SmartScreen.** The binaries are not code signed yet, so the first
> launch shows "Windows protected your PC". Click **More info → Run anyway**.
> To check a download, compare its SHA-256 with `SHA256SUMS.txt` on the
> release: `Get-FileHash .\houwritr-windows-<arch>-vX.Y.Z-setup.exe`.

On first use the selected engine downloads its verified model files into
`%LOCALAPPDATA%\HouWritr\models\`: Parakeet CPU about 670 MB, Parakeet NPU
about 650 MB more, Whisper NPU a 2.0 GB archive (about 2.2 GB unpacked).
Progress shows in the tray.

## Using HouWritr

HouWritr lives in the system tray. Right click the icon:

| Tray menu | What it does |
|---|---|
| **Transcribe audio or video file…** | Pick a file; a window shows progress and the result, with **Stop** while it runs (nothing is saved). |
| **Start / Stop meeting recording** | Record your microphone and the PC's sound, transcribed while you record. |
| **Pause / Resume meeting recording** | Stop capturing for a while without ending the meeting. |
| **Hide / Show recording indicator** | Toggle the meeting pill for this session only. |
| **Jobs…** | Everything running or waiting: the meeting being recorded, files and meeting parts with their progress, and background cleanup retries with their next try. Files can be stopped and retries cancelled from here. |
| **Settings** | Engine, speaker labels, AI cleanup, output format, startup. |
| **Open logs / Export diagnostics…** | See [Logs and diagnostics](#logs-and-diagnostics). |
| **Retry model / engine, Cancel model download** | Recover from a failed or unwanted download. |
| **About, credits & support…** | Version, licences, links. |

Transcripts are always saved as files. HouWritr never pastes text and never
puts anything on the clipboard. A small dark pill at the bottom centre of the
main screen shows queued work, transcription, AI cleanup and short notices
(it can be turned off in Settings). Settings take effect as soon as they are
saved, with no restart.

## Transcribing a file

**Transcribe audio or video file…** accepts WAV, MP3, M4A/AAC, FLAC, OGG and
MP4/M4V/MOV/MKV with a supported audio track. A progress window opens with:

* the stage (reading audio, finding speakers, transcribing n of m, AI
  cleanup), elapsed time and an estimate;
* the source, its length, the engine, whether AI cleanup is on, and the
  output file;
* when finished: the word count (Chinese and Japanese characters count one
  each), a preview, and **Open file**, **Show in folder** and **Close**.

Minimise it like any window; closing it does not cancel the job. The
transcript is saved next to the source as `<name>.txt` (or `.md`, see
[Settings](#settings)); an existing file is never overwritten, a number is
added instead.

Long audio is cut into windows of at most 30 seconds at the quietest nearby
moment, so words are not split. With Whisper, a window the model cannot
finish (it starts repeating itself, or runs out of its 199 decoding steps on
fast speech) is decoded again in two halves, and any repetition loop is cut
to a single copy.

## Recording a meeting

**Start meeting recording** captures your microphone and everything playing
on the default output device (WASAPI loopback) into one timeline, up to 12
hours. It works for online calls and for meetings in a room (with nothing
playing, only the microphone contributes).

* **Pause** stops capturing and releases the microphone (Windows' microphone
  indicator turns off). **Resume** continues the same recording; paused time
  is not recorded or counted.
* **Rolling transcription.** Every five minutes of audio, and whenever you
  pause, the finished part is cut off at a quiet moment and transcribed in the
  background while recording continues. Each part is appended to
  `Meeting <date> <time> (in progress).txt`, so you can read the meeting so
  far and nothing is lost if the app or the PC stops. When you stop, only the
  last few minutes are left to transcribe; the parts are joined into the final
  transcript and the in progress file is deleted. Memory use stays around five
  minutes of audio however long the meeting runs.
* Speaker labels and `[m:ss]` markers stay consistent across parts (Speaker 1
  in the first part is Speaker 1 in the last; times count from the start of
  the meeting). A part that fails to transcribe leaves a visible gap and the
  rest is still saved.
* **Recording indicator.** A pill shows red `Recording 12:34` with live level
  bars, or amber `Paused 12:34`. It steps aside while the status pill needs the
  same spot. Turn it off in Settings, or for the session from the tray. The
  tray icon turns red (recording) or amber (paused) either way.
* Transcripts are saved to `Documents\HouWritr\Meetings\Meeting <date>
  <time>.txt`, named after when the meeting started. Quitting while recording
  saves what was recorded.

For in person meetings, use the best microphone available (a conference or
USB microphone beats a laptop's), keep it near the speakers, and expect quiet
or distant voices and overlapping speech to be missed.

Record meetings only where everyone present has agreed to it.

**Turning a meeting into a report.** [`docs/meeting-report-prompt.md`](docs/meeting-report-prompt.md)
is a ready made prompt for Claude: give it the saved transcript plus the
agenda and slides, and it builds a single HTML report with a summary,
decisions, action items, a topic digest and the full transcript with an index.

## Speaker labels

Turn on *Label speakers in file and meeting transcripts* (Settings → Speaker
labels) and transcripts are split by who is talking:

```
[0:00] Speaker 1: Thanks everyone for joining…

[0:12] Speaker 2: Happy to be here. Can we start with the roadmap?
```

`[m:ss]` is the time into the recording where that person starts talking.

Two small models (pyannote segmentation 3.0 and a WeSpeaker voice print model,
about 32 MB, downloaded once) find the speaker turns on the CPU, roughly a
minute or two per hour of audio; each turn is then transcribed separately.
Labels never remove speech: audio that cannot be attributed is transcribed with
the neighbouring speaker. Labels are numbered in order of appearance and are
**not** names. Accuracy drops with overlapping speech, short interjections,
similar voices and poor audio, and one voice is sometimes split into two
labels. To limit that, a voice heard for less than 30 seconds (or 2% of all the
speech, if that is more) is counted as the closest voice that was heard enough,
and in a meeting this is decided once, after Stop, across all the parts: applause or
a stray word no longer becomes a "Speaker 17". The cost is that someone who only
speaks briefly is labelled as another speaker. With a single speaker the transcript stays unlabelled; if the models
cannot run you still get the transcript and a note saying so.

The Speaker labels section also shows whether the models are installed, with
**Download and install**, **Verify files** (sizes and SHA-256), **Open model
folder** and **Remove models**, plus an *Install manually* list (names, sizes, hashes, links)
for machines without direct internet access.

## AI cleanup and meeting notes

### Cleanup

With *Text enhancement* **On**, each file and meeting transcript is sent, as
text and never as audio, to the provider you chose, which fixes punctuation
and recognition slips:

* The transcript is sent in slices: about 3,000 characters for a hosted
  provider, 1,200 for a local server and 600 for the built in NPU model (local
  models get a third of that for Chinese and Japanese, which take more tokens
  per character). Slices are cut at sentence ends and never split a speaker's
  block.
* With a hosted provider the first slice goes alone, then three slices are
  cleaned at a time; the results are put back in order. A model on this PC
  gets one slice at a time.
* `[m:ss]` markers are held back from the request and put back afterwards.
* If most of the Chinese in a transcript is Simplified (or Traditional), every
  slice is told to write in that script, so a transcript does not drift
  between the two.
* Each cleaned slice is checked by an integrity validator (numbers, negations,
  acronyms, URLs, language, length). A slice that fails keeps its recognised
  text, and the log says why. If no slice can be cleaned, the raw transcript is
  saved with a warning that names the reason.

**Changes the safety checks refuse.** When the cleaned version of a section
changes a number or a speaker label, that section is not discarded at once:
it is cut in two at the sentence or block boundary nearest its middle, each
half is cleaned on its own (and cut again if it fails too, down to about a
quarter of the section), and the pieces are put back together with the gaps
they had. Only the smallest piece that keeps failing keeps its recognised
text. The transcript window counts such sections as cleaned in part.

**Unreliable connections.** Cleanup never holds up a transcript. A slice that
cannot reach the provider keeps its recognised text; after three failures in a
row the rest is skipped, and for a minute no further requests are made.
Unfinished sections of a meeting get a second try when you press Stop (up to
90 seconds). Anything still unfinished is retried in the background 2, 5, 10,
20 and 30 minutes after saving, and the file is updated in place, unless you
have edited, moved or deleted it. The transcript window shows each retry and
its progress, and says "Done" only when nothing is left to retry. Timeouts
grow with the slice length (up to 10 minutes), for slow networks.

Leave enhancement **Off** to keep everything on this PC. With it on, the
transcript text goes to your provider, so think twice for confidential
meetings.

### Providers

* **GitHub Copilot.** Uses the account GitHub Copilot CLI is signed in with
  (`winget install GitHub.Copilot`, then `copilot login`), found in the order
  the CLI uses: the `COPILOT_GITHUB_TOKEN`, `GH_TOKEN` or `GITHUB_TOKEN`
  variable, its Windows Credential Manager entry, its config file, and finally
  the GitHub CLI (`gh auth login`). The model list in Settings comes from
  Copilot for your account (**Refresh** asks again); Copilot Free has no
  fixed list, as GitHub offers it models through automatic selection only.
  If Copilot refuses the chosen model for
  your account, HouWritr asks Copilot which models are on offer and tries the
  closest few under both the Copilot CLI and VS Code identities, then keeps
  using the one that worked for the rest of the session. The log shows each
  model's policy and every attempt; a model whose policy is `disabled` has to
  be enabled in your GitHub Copilot settings, or pick another.
* **OpenAI compatible API.** Needs a base URL, model ID and API key. The key is
  kept in Windows Credential Manager, never in the settings file, and is only
  sent over https (or to a server on this PC).
* **Local model (on this PC).**
  * *Built in model*: on a Snapdragon PC, HouWritr uses Qualcomm's
    [GenieX](https://github.com/qualcomm/GenieX) to run a model for you.
    Press **Install GenieX…** (the pinned v0.8.0 installer, checked against
    its SHA-256, installed for your user only), pick a model, then
    **Download model** (one bar with size, speed and time left; a download
    that stops resumes by itself, up to five tries) and **Test model**, which shows how fast
    it writes. One Qwen3 4B file (`unsloth/Qwen3-4B-Instruct-2507-GGUF:Q4_0`,
    about 2.4 GB from Hugging Face), run two ways through GenieX's llama.cpp
    runtime:
    * *Qwen3 4B in hybrid mode* (the default): GenieX's `hybrid` setting, where
      llama.cpp's scheduler chooses per step between the NPU and the CPU
      (Qualcomm's documentation; GenieX does not log which one ran a step).
      The faster choice (about 18 tokens a second on a Snapdragon X Elite);
      on the test PC the CPU stayed below 50% with other work in the
      foreground.
    * *Qwen3 4B on the NPU only*: all layers on the NPU. Steady (about 13
      tokens a second) and the lightest on the CPU.

    Earlier versions also offered Qualcomm's precompiled NPU bundle (about
    one token a second on the test PC), the CPU alone, and Qwen2.5 7B. Their
    files are not deleted by the update; **Uninstall GenieX…** removes them.

    HouWritr starts `geniex serve` on a private port (18651) only when a
    job needs it and stops it as soon as that job is finished (each meeting
    part, each background retry and each test in Settings included), or
    when it stops answering or slows down sharply. If a model
    writes too slowly to be useful (a slice takes over two minutes), cleanup
    stops and the rest of the transcript stays as recognised, without
    background retries. **Remove model** deletes the selected model;
    **Uninstall GenieX…** removes all of them and runs GenieX's own
    uninstaller.
  * *Another local server*: any OpenAI compatible server on `localhost`
    (LM Studio, Ollama, Foundry Local, your own GenieX). Presets, **Find
    models**, **Test connection** and a setup guide are in Settings.

  With a model on this PC, a meeting is cleaned up once, after you press
  Stop: while it runs, its parts are only transcribed. The model then loads
  once instead of every five minutes, and stays off the CPU during the call.
  Hosted providers clean each part as it comes, so their result is ready at
  Stop.

  Local models are slower and weaker than hosted ones, so they get smaller
  slices, longer waits, and `<think>…</think>` blocks removed from replies.
  The same integrity checks apply. GenieX is a Qualcomm developer preview.

Cleanup prompts can be customised per provider, endpoint and model; the
transcript always travels as a separate, untrusted message that the model must
not answer or obey.

### Meeting notes

Under *Settings → Output and feedback* choose what happens when a meeting
stops: **Ask every time** (default), **Always** or **Never**. Notes are saved
next to the transcript as `Meeting notes <date> <time>.md` with *Summary*, *Key
points*, *Decisions*, *Action items* (a `- [ ]` checklist) and *Open
questions*. They need a provider, work even with cleanup off (with cleanup on
they are built from the cleaned text), and very long meetings are noted in
parts and merged. The prompt forbids inventing names, owners, dates or
decisions, but review notes before acting on them.

## Settings

Tray icon → right click → **Settings**:

* **Transcription**: Parakeet CPU INT8 (default), Parakeet NPU, or Whisper
  Large v3 Turbo NPU, each with hardware availability, download size and cache
  status. NPU engines are shown but disabled unless the ARM64 build runs on a
  Snapdragon X Elite. **Remove model files** deletes a downloaded engine you
  are not using (it downloads again if you select it later).
* **Output and feedback**: save transcripts as Markdown (`.md` with a title,
  details line and paragraphs; added by HouWritr, not the AI) or plain text;
  show the status pill; show the meeting recording indicator; and what to do
  about meeting notes.
* **Speaker labels**: on or off, model status and management, and **Remove
  models** (see above).
* **Text enhancement**: **Off** or **On**, the provider, model and prompt.
* **Startup**: start HouWritr when you sign in.
* **About, credits & support**: version, architecture, source, issues,
  privacy, licences, model and runtime credits.

Settings are stored in `%LOCALAPPDATA%\HouWritr\settings.json`, written
atomically and picked up live, including edits made outside the app. Settings
files from older (dictation) versions still load; their shortcut and paste
options are ignored, and cleanup that used to need Shift is simply on.

## Files and folders

| What | Where |
|---|---|
| File transcripts | Next to the source file |
| Meeting transcripts and notes | `Documents\HouWritr\Meetings\` |
| Transcripts that could not be saved where they belong | `Documents\HouWritr\` |
| Settings, models, logs | `%LOCALAPPDATA%\HouWritr\` |
| API keys | Windows Credential Manager |

The data folders were called OpenWritr before the rename. HouWritr moves them
to the new name the first time it starts, with models and settings intact (the Credential Manager entry for an API key
and the startup entry move too). `AppData` is hidden; paste
`%LOCALAPPDATA%\HouWritr` into the Explorer address bar to open it.

## Logs and diagnostics

Logs are in `%LOCALAPPDATA%\HouWritr\logs`:

* `sessions\` holds one file per run of the app (the settings and transcript
  windows it opens write into the same file).
* `recordings\` holds one file per file transcript, meeting or cleanup retry,
  with every line logged while it was worked on. All parts of a meeting share
  one file.
* File names start with the start time in **UTC**, matching the times inside.
  The newest 30 sessions and 300 recordings are kept.
* The first line of each session shows the app version and `build=<commit>`,
  so you can tell which build ran.
* `crash.log` records any crash with its backtrace.

**Open logs** opens the folder. **Export diagnostics…** builds a bounded zip
with recent logs (sanitised) and redacted settings, runtime and model status;
it never contains audio, transcript text or API keys.

## Architecture

HouWritr is a native Rust app: no Electron, no web view, no Python at run
time.

```
HouWritr/
├── src/
│   ├── main.rs              entry; tray, `--settings`, `--transcript-window` modes, logging
│   ├── app.rs               winit event loop, tray menu, file and meeting jobs, repairs
│   ├── worker.rs            inference thread: diarisation, transcription, cleanup
│   ├── meeting.rs           microphone + loopback capture, rolling parts, file names
│   ├── media.rs             file decoding (Symphonia), resampling, quiet point search
│   ├── transcript_window.rs progress and result window (subprocess)
│   ├── overlay.rs           status pill and meeting pill (layered Win32 window, GDI)
│   ├── tray.rs              tray icon and menu
│   ├── settings.rs          validation, migration, atomic JSON persistence
│   ├── settings_ui.rs       eframe/egui settings dialog (subprocess)
│   ├── asr/                 Parakeet (CPU/NPU), Whisper NPU, speaker diarisation
│   ├── enhance.rs           cleanup and notes: slicing, parallel requests, retries
│   ├── cleanup/             prompts, request building, integrity validator
│   ├── copilot_auth.rs      GitHub Copilot sign-in lookup
│   ├── local_llm.rs         GenieX install, model download, private `geniex serve`
│   ├── log_files.rs         session and recording logs
│   ├── diagnostics.rs       log pruning, diagnostics export, Explorer helpers
│   └── bin/                 `package` (release zip) and `cleanup_eval`
└── Cargo.toml
```

Key design decisions:

* **Work off the tray thread.** File decoding, transcription, diarisation and
  cleanup run on worker threads; the tray loop only routes events, so the menu
  stays responsive during a two hour transcription.
* **Windows as subprocesses.** Settings, About and the transcript window are
  the same exe relaunched with a flag, spawned from a background thread
  (`CreateProcessW` on Windows ARM64 with Defender can block for seconds), so
  a slow or crashing window can never stall meeting capture.
* **Pills on their own message loop.** The status and meeting pills are
  layered top most windows painted with GDI on their own threads; they read
  the meeting level from an atomic and take state through a channel.
* **Multi channel downmix.** The Qualcomm Aqstic microphone array on Surface
  Pro exposes 4 to 8 channels at 48 kHz; they are averaged to mono before
  resampling to 16 kHz.
* **Power.** HouWritr and its GenieX server opt out of Windows power
  throttling, so long background jobs are not slowed when no window is in
  front.

## NPU pipeline

Parakeet runs its pre-compiled encoder on the Hexagon HTP; mel preprocessing
and the TDT decoder stay on CPU. Whisper runs both its encoder and
autoregressive KV-cache decoder as pre-compiled QNN sessions. It processes
sequential 30-second chunks, detects language once per recording, and reuses
that language for every later chunk. A chunk that loops or runs out of
decoding steps is decoded again in two halves.

The compiled binary expects a fixed **8-second audio window**. For audio of
8 s or less the encoder is run once on a padded window.
For longer audio the encoder is run in chunks (8 s window with 1 s
overlap), feature streams are stitched at the seam, and the TDT decoder
runs once over the concatenated features. Tested up to ~23 s without
boundary doubling.

### Measured on Snapdragon X Elite (X1E80100)

| Audio length | Decode (preproc + encode + TDT) | × Realtime | Chunks |
|---|---|---|---|
| 3 s | 128 ms | 23× | 1 |
| 5.8 s | 221 ms | 26× | 1 |
| 16.4 s | 375 ms | 44× | 3 |
| 23.0 s | 626 ms | 37× | 4 |

The encoder itself is ~67 ms ± 0 ms per 8-second window, independent of
the actual audio length within the window.

### How the binary was built

1. `scripts/build_npu_encoder.py` constant-folds the encoder's
   dynamic-shape attention mask (`Shape → Gather → Range → Expand`) against
   a frozen `[1, 128, 801]` input — the HTP backend cannot evaluate that
   subgraph as-is.
2. `scripts/aihub_compile_encoder.py` submits the static-shape FP32 ONNX
   plus FLEURS calibration samples to Qualcomm AI Hub. Quantize job uses
   INT8 weights / INT16 activations (the standard HTP recipe for
   transformer encoders); compile job targets `snapdragon_x_elite_crd`
   with `--target_runtime qnn_context_binary --truncate_64bit_io`.
3. `scripts/wrap_qnn_context_binary.py` wraps the resulting `.bin` in a
   408-byte EPContext-node ONNX so ORT's QNN EP can consume it.
4. `src/asr/qnn_ffi.rs` loads the wrapper via direct `ort_sys` C-API calls,
   bypassing `ort` 2.0-rc.12's session builders (which crash inside QnnHtp
   when consuming EPContext-wrapper ONNX).

### Required helper DLLs

The QNN backend loads several sibling DLLs by name at session-create time.
`scripts/fetch_runtime.py` extracts the exact, hash-pinned set recorded in
`runtime-manifest.json`. Alongside `onnxruntime_providers_qnn.dll`, the
X Elite path requires `QnnHtp.dll`, `QnnHtpPrepare.dll`,
`QnnHtpNetRunExtensions.dll`, `QnnSystem.dll`, the V73 stub and skeleton,
and its catalog file. Without the SKEL + `.cat` pair, the stub fails
`LoadLibrary` with `ERROR_MOD_NOT_FOUND` (126) and QnnHtp later aborts
session creation with `STATUS_STACK_BUFFER_OVERRUN` (0xC0000409) without
a useful error.

## Models

| Model | Provider | Size | License | Auto-downloaded? |
|---|---|---|---|---|
| Parakeet TDT 0.6B v3 (CPU INT8 ONNX + companion files) | [istupakov/parakeet-tdt-0.6b-v3-onnx](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx) | ~670 MB | CC-BY-4.0 | Yes, on first run |
| Parakeet TDT 0.6B v3 NPU encoder (QAIRT context binary + wrapper) | [trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s](https://huggingface.co/trsdn/parakeet-tdt-0.6b-v3-htp-int8-8s) | ~632 MB | CC-BY-4.0 | Yes, on first NPU launch |
| Whisper Large v3 Turbo (QNN encoder, decoder, tokenizer) | [qualcomm/Whisper-Large-V3-Turbo](https://huggingface.co/qualcomm/Whisper-Large-V3-Turbo) | ~2.2 GB extracted | Apache 2.0 + BSD-3 | Yes, on first Whisper launch |

The NPU encoder model is device-gated to Snapdragon X Elite (Hexagon V73).
It will not run on X Plus or any other Qualcomm chipset without
recompilation via AI Hub.

## Licenses

- **HouWritr code**: MIT — see [`LICENSE`](LICENSE).
- **Parakeet model**: CC-BY-4.0 (NVIDIA). Attribution preserved when the
  model is downloaded.
- **Qualcomm QNN runtime DLLs** (`QnnHtp.dll`, QNN provider, V73 stub, etc.,
  bundled in the release zip): **Qualcomm AI Engine Direct redistributable
  license**. The full text ships inside every release zip under
  `third-party-licenses/Qualcomm_LICENSE.pdf`, alongside Microsoft's
  `ThirdPartyNotices.txt` for the `onnxruntime-qnn` PyPI package the DLLs
  come from. These DLLs are redistributable as part of applications targeting
  Qualcomm Snapdragon hardware, which is what HouWritr does.
- **ONNX Runtime DLLs** (`onnxruntime.dll`, `onnxruntime_providers_qnn.dll`):
  MIT (Microsoft), bundled under their respective LICENSE files in the release
  zip.

## Repository layout

```
HouWritr/
├── src/                   Rust native app
├── scripts/               pinned runtime and release tooling
├── models-manifest.json   pinned model assets and hashes
├── runtime-manifest.json  pinned ORT/QNN wheels and compatibility tuple
├── release-manifest.json  canonical per-architecture package contents
└── target/                gitignored build and staging output
```

The shipped `houwritr.exe` does not call Python at runtime.

## Development

```powershell
git clone https://github.com/raulhou/HouWritr.git
cd HouWritr
.\scripts\envup.ps1
cargo build --release --bins
python scripts/fetch_runtime.py --arch arm64
python scripts/prepare_release.py --arch arm64
target\stage\arm64\houwritr.exe --self-check
```

For x64 packaging, stage the CPU runtime with:

```powershell
python scripts/fetch_runtime.py --arch x64
```

See [`docs/RUNTIME_COMPATIBILITY.md`](docs/RUNTIME_COMPATIBILITY.md) for
the shared ORT/QNN/QAIRT contract and its Snapdragon hardware gates.

### Cleanup evaluator (`cleanup_eval`)

`src/bin/cleanup_eval/` is a focused, opt-in evaluator for the `cleanup`
core (prompt resolution, request/response handling, normalization, and the
integrity validator) — it reuses those production modules directly rather
than reimplementing them. It replays a committed, versioned, fully
synthetic DE/EN fixture corpus
([`fixtures/cleanup-eval/v1/corpus.json`](fixtures/cleanup-eval/v1/corpus.json))
covering punctuation, fillers, repetition, negation, digits/dates/times/
versions, commands/code/acronyms, URLs/emails, language preservation,
filler-only/`[[EMPTY]]`, and known tricky cases. No private recordings are
included or ever required.

```powershell
# Deterministic offline mode (default): no network, safe for CI.
cargo run --bin cleanup_eval
cargo run --bin cleanup_eval -- --json --category punctuation

# Focused tests: corpus versioning, expected decisions, no private fields
# in the serialized report, and production-module reuse.
cargo test --bin cleanup_eval
```

Live-provider mode is entirely opt-in (`--live`) and never required by tests
or releases: it reads the same `settings.json` provider/model/base URL and
the same credential storage as `gh auth token` (the app itself also reads the Copilot CLI sign-in) or Windows
Credential Manager), but never prints secrets, transcripts, prompts, or raw
candidate text. Every report — offline or live — includes the model/
provider, canonical endpoint scope, prompt-profile/validator/corpus
version, latency, failure/integrity category counts, and punctuation/
filler/repetition health scores, with all content redacted by default:

```powershell
cargo run --bin cleanup_eval -- --live --provider github_copilot
```

Run `cargo run --bin cleanup_eval -- --help` for the full flag list.

## Releases

Tagged releases live at
[github.com/raulhou/HouWritr/releases](https://github.com/raulhou/HouWritr/releases).
Each architecture ships a ZIP and installer generated from the same validated
release stage. CI also builds an unsigned Store `.msixbundle` as a workflow
artifact for Partner Center ingestion, but does not attach it to the public
GitHub release. Tagged builds use the repository variables
`MSIX_IDENTITY_NAME` and `MSIX_PUBLISHER` when configured; otherwise the Store
bundle is skipped rather than publishing a placeholder identity. Every package
includes an `artifact-manifest.json`; CI verifies all required file hashes and
confirms that x64 artifacts contain no Qualcomm runtime files.
