# Microsoft Store listing text

Text to paste into Partner Center for "HouWritr for Windows". The package
manifest (`installer/AppxManifest.template.xml`) carries the short form; keep
the two in step. Everything below describes what the app does today.

## Short description (up to 100 characters)

Record meetings and transcribe audio and video files on your own PC.

## Description

HouWritr for Windows records your meetings and turns audio and video files into
text. Speech recognition runs on your PC: on the Snapdragon NPU on Copilot+ PCs
with a Snapdragon X chip, or on the CPU on other PCs.

Meeting recording
* Records your microphone and the sound playing on the PC into one timeline,
  and transcribes while you record.
* Pause, resume and stop; the transcript is saved as you go.
* Optional meeting notes written from the transcript.

File transcription
* Drop in an audio or video file and get a transcript saved next to it.
* A Jobs window shows what is running and waiting, and every job can be stopped.

Speaker labels
* Optional "Speaker 1", "Speaker 2" labels, also worked out on your PC.

AI cleanup (optional, off by default)
* Fixes punctuation and obvious recognition mistakes without changing numbers
  or who said what, and keeps the original text for any section it cannot
  clean safely.
* Runs with a model on this PC, with GitHub Copilot, or with an OpenAI
  compatible service you set up yourself. Text is only sent to a service when
  you choose one.

Privacy
* No telemetry and no account. Recordings and transcripts stay on your PC.
* Downloaded speech models and settings are stored on your PC.

## Search terms

meeting recorder, transcription, speech to text, Snapdragon NPU, transcript,
speaker labels

## Notes for certification

* The app records the microphone and the PC's output sound (WASAPI loopback)
  only while a meeting recording is running, and shows that it is recording.
* `runFullTrust` is needed because the app is a classic Win32 program: it uses
  native speech recognition libraries (ONNX Runtime with Qualcomm QNN), writes
  transcripts to folders the user picks, and starts a local helper process.
* On first use the app downloads speech models from Hugging Face. If the user
  chooses the optional model on this PC, it also downloads Qualcomm's GenieX
  tool and a model from Hugging Face. Nothing is run from a download without
  the user asking for that feature.
* Testing: transcribing a file works on any PC. The NPU engines need a
  Snapdragon X PC; on other PCs the CPU engine is used.
