# Live Transcriber

A small Windows desktop app that transcribes whatever the PC is playing (a call, a lecture, a video), optionally mixed with your microphone, and writes the transcript to a text file as it goes. Everything runs locally.

The speech recognition is done by OpenAI's Whisper models through the faster-whisper library. The models and the library are not mine. This project is the app around them: audio capture, mixing, chunking, the transcription queue and the UI.

## How it works

1. **Capture.** The default speakers are recorded through WASAPI loopback (PyAudioWPatch). If the mic is on, a second stream records the default microphone.
2. **Mix.** Both streams are resampled to 16 kHz mono and added onto one shared timeline, using each packet's timestamp so the two stay in sync. Long silences are skipped.
3. **Chunk.** Every so often the buffered audio is cut into a piece for the model. A piece is sent when the sound pauses for 1.2 s, when it reaches 8 s and ends quietly, at least every 12 s if the sound never stops, and never longer than 30 s. Cuts are placed at the quietest 0.2 s window near the end so words are not split.
4. **Transcribe.** A background thread loads the Whisper model once and transcribes pieces in order. It uses voice activity detection to skip non-speech, word timestamps, and the end of the previous text as a prompt to keep names and punctuation consistent. Quiet audio is normalized first.
5. **Stitch.** When a piece was cut mid-sound, the words in its last 2.5 s are held back and re-transcribed together with the next piece, so sentences that straddle a cut come out whole.
6. **Write.** Each line is shown in the window and appended to `Transcript <date> <time>.txt` immediately, so a crash does not lose the transcript.

On a machine with an NVIDIA GPU it runs `large-v3-turbo` on CUDA (float16). Otherwise, or if CUDA fails to load, it falls back to `small` on the CPU (int8).

## Requirements

- Windows 10 or 11 (WASAPI loopback and the shortcut helper are Windows-only)
- Python 3 (installed with the `py` launcher, which `setup.bat` uses)
- Optional: an NVIDIA GPU with a recent driver

## Setup

```bat
setup.bat
```

This creates `.venv`, installs `requirements.txt` (plus the CUDA libraries in `requirements-gpu.txt` if `nvidia-smi` is found) and adds Desktop and Start menu shortcuts. The speech model is downloaded from Hugging Face on first launch.

To run it without the shortcut:

```bat
.venv\Scripts\python transcriber.py
```

## Usage

Press **Start** to record and **Stop** to finish. **Mic on / Mic off** adds the default microphone (use headphones on calls so the mic does not pick up the speakers). **Change folder** sets where transcripts are saved; the default is `Documents\Transcripts`.

## Settings

`config.json` is created next to the script:

| Key | Meaning |
|---|---|
| `folder` | Where transcripts are saved |
| `model` | Whisper model name, e.g. `small`, `medium`, `large-v3-turbo`. Empty picks automatically |
| `language` | Language code such as `en`. Empty detects it |
| `device` | `cpu` to skip the GPU |
| `mic`, `mic_device` | Mic on/off, and part of a mic's name to use instead of the default |

Errors are written to `transcriber.log`.

## Files

```
transcriber.py      the app (capture, mixing, chunking, transcription thread, Tkinter UI)
make_shortcut.py    creates the Desktop and Start menu shortcuts
setup.bat           one-time setup
requirements.txt    faster-whisper, PyAudioWPatch, numpy, pywin32
requirements-gpu.txt  CUDA libraries (cuBLAS, cuDNN) for GPU inference
```

## Third-party models and libraries

| Component | By | License | Link |
|---|---|---|---|
| Whisper speech recognition models | OpenAI | MIT | [github.com/openai/whisper](https://github.com/openai/whisper), [model card](https://github.com/openai/whisper/blob/main/model-card.md) |
| faster-whisper (CTranslate2 inference) | SYSTRAN | MIT | [github.com/SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| `small` model weights (CTranslate2 conversion) | SYSTRAN | MIT | [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small) |
| `large-v3-turbo` weights (CTranslate2 conversion) | Mobius Labs | MIT | [mobiuslabsgmbh/faster-whisper-large-v3-turbo](https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo) |
| Silero VAD (bundled with faster-whisper) | Silero Team | MIT | [github.com/snakers4/silero-vad](https://github.com/snakers4/silero-vad) |
| PyAudioWPatch (WASAPI loopback) | s0d3s | Apache-2.0 | [github.com/s0d3s/PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch) |

No model weights are included in this repository; faster-whisper downloads them on first use.

## Limitations

Windows only. Transcription lags a few seconds behind the audio. Accuracy depends on the model: the CPU fallback (`small`) is noticeably less accurate than `large-v3-turbo`. There is no speaker labeling.
