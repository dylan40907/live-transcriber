"""Live Transcriber: records what the PC is playing and writes a local transcript.

Start records the default speakers (WASAPI loopback). Speech is transcribed
on this PC with faster-whisper while it records. Stop finishes the last bit
and leaves the transcript as a .txt in the chosen folder.
"""

import ctypes
import glob
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
ICON_PATH = os.path.join(HERE, "transcriber.ico")
LOG_PATH = os.path.join(HERE, "transcriber.log")

TARGET_RATE = 16000
MAX_CHUNK_S = 30.0      # longest piece sent to whisper at once
MIN_CHUNK_S = 8.0       # shortest piece cut at a pause while sound keeps coming
QUIET_GAP_S = 1.2       # no audio packets this long = flush what we have
LIVE_CHUNK_S = 12.0     # with nonstop sound, send a piece at least this often
HOLD_MAX_S = 25.0       # never hold back audio once a piece is this long
HOLD_TAIL_S = 2.5       # words in the last this-many seconds of a cut piece are redone next time

BG = "#ffffff"
FG = "#000000"
MUTED = "#6b6b6b"
LINE = "#000000"
HOVER = "#2b2b2b"
REC = "#d40000"
FONT = "Segoe UI"


def log(*parts):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(p) for p in parts) + "\n")
    except Exception:
        pass


def load_config():
    cfg = {"folder": "", "model": "", "language": "", "mic": True, "mic_device": ""}
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg.update(json.load(f))
    except Exception:
        pass
    if not cfg["folder"]:
        cfg["folder"] = os.path.join(os.path.expanduser("~"), "Documents", "Transcripts")
    return cfg


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        log("config save failed:", e)


def add_cuda_dll_dirs():
    """cuBLAS / cuDNN from the nvidia-* pip wheels, if they are installed."""
    for base in sys.path:
        for bin_dir in glob.glob(os.path.join(base, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(bin_dir)
            except Exception:
                pass
            os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")


def resample(x, rate):
    if rate == TARGET_RATE or len(x) == 0:
        return x.astype(np.float32)
    if rate % TARGET_RATE == 0:
        k = rate // TARGET_RATE
        n = len(x) // k * k
        return x[:n].reshape(-1, k).mean(axis=1).astype(np.float32)
    width = max(1, int(round(rate / TARGET_RATE)))
    if width > 1:
        x = np.convolve(x, np.ones(width, dtype=np.float32) / width, mode="same")
    n_out = int(len(x) * TARGET_RATE / rate)
    pos = np.arange(n_out, dtype=np.float64) * (rate / TARGET_RATE)
    return np.interp(pos, np.arange(len(x)), x).astype(np.float32)


# ---------------------------------------------------------------- recording

class StreamResampler:
    """Turns a stream of mono blocks at `rate` into 16 kHz blocks, without seams."""

    def __init__(self, rate):
        self.rate = rate
        self.step = rate / TARGET_RATE
        self.k = int(round(self.step))
        self.whole = abs(self.step - self.k) < 1e-9
        self.pending = np.zeros(0, np.float32)
        self.pos = 0.0

    def __call__(self, x):
        x = np.concatenate([self.pending, x])
        if self.whole:
            n = len(x) // self.k * self.k
            self.pending = x[n:]
            return x[:n].reshape(-1, self.k).mean(axis=1).astype(np.float32)
        if len(x) < 2:
            self.pending = x
            return np.zeros(0, np.float32)
        if self.k > 1:
            x = np.convolve(x, np.ones(self.k, np.float32) / self.k, mode="same")
        last = len(x) - 1
        n = int((last - self.pos) // self.step) + 1 if last >= self.pos else 0
        where = self.pos + np.arange(n) * self.step
        out = np.interp(where, np.arange(len(x)), x).astype(np.float32)
        nxt = self.pos + n * self.step
        keep = max(0, int(nxt) - 1)
        self.pending = x[keep:]
        self.pos = nxt - keep
        return out


PA_LOCK = threading.Lock()  # PortAudio can't start or stop in two threads at once


class Source:
    """One capture stream: the speakers (loopback) or the microphone."""

    def __init__(self, recorder, kind, hint=""):
        self.recorder = recorder
        self.kind = kind            # "speakers" | "mic"
        self.hint = hint.lower()
        self.next = None            # where its next block goes in the mix
        self.error = ""
        self.resampler = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _pick(self, pa):
        import pyaudiowpatch as pyaudio
        wasapi = pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        if self.kind == "mic":
            if self.hint:
                for i in range(pa.get_device_count()):
                    dev = pa.get_device_info_by_index(i)
                    if (dev["hostApi"] == wasapi["index"] and dev["maxInputChannels"] > 0
                            and not dev.get("isLoopbackDevice") and self.hint in dev["name"].lower()):
                        return dev
            if wasapi["defaultInputDevice"] < 0:
                raise RuntimeError("No microphone found")
            return pa.get_device_info_by_index(wasapi["defaultInputDevice"])
        try:
            return pa.get_default_wasapi_loopback()
        except Exception:
            speakers = pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
            for dev in pa.get_loopback_device_info_generator():
                if speakers["name"] in dev["name"]:
                    return dev
            raise RuntimeError("No speakers found")

    def _push(self, data, ch):
        now = time.time()
        a = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        if ch > 1:
            a = a[: len(a) // ch * ch].reshape(-1, ch).mean(axis=1)
        dur = len(a) / self.resampler.rate
        self.recorder.add(self, self.resampler(a), now - dur)

    def _run(self):
        import pyaudiowpatch as pyaudio
        # PortAudio only sets up COM on the thread that starts it; WASAPI needs it here too.
        ctypes.windll.ole32.CoInitialize(None)
        rec = self.recorder
        while rec.running:
            with PA_LOCK:
                pa = pyaudio.PyAudio()
            stream = None
            try:
                dev = self._pick(pa)
                ch = max(1, int(dev["maxInputChannels"]))
                rate = int(dev["defaultSampleRate"])
                self.resampler = StreamResampler(rate)
                self.next = None

                def callback(data, frames, info, status):
                    self._push(data, ch)
                    return (None, pyaudio.paContinue)

                stream = pa.open(format=pyaudio.paInt16, channels=ch, rate=rate, input=True,
                                 input_device_index=dev["index"], frames_per_buffer=rate // 50,
                                 stream_callback=callback)
                stream.start_stream()
                self.error = ""
                name = dev["name"]
                log("capturing", self.kind, name, rate, "Hz", ch, "ch")
                # Reopen when the default device changes (e.g. headphones) or the stream dies.
                while rec.running and stream.is_active():
                    for _ in range(30):
                        if not rec.running:
                            break
                        time.sleep(0.1)
                    if not rec.running:
                        break
                    if self.hint:
                        continue
                    with PA_LOCK:
                        check = pyaudio.PyAudio()
                    try:
                        now = self._pick(check)["name"]
                    except Exception:
                        now = name
                    finally:
                        with PA_LOCK:
                            check.terminate()
                    if now != name:
                        log("default", self.kind, "changed to", now)
                        break
            except Exception as e:
                self.error = str(e) or "failed"
                log(self.kind, "capture error:", e)
                time.sleep(1)
            finally:
                try:
                    if stream:
                        stream.stop_stream()
                        stream.close()
                except Exception:
                    pass
                with PA_LOCK:
                    pa.terminate()


class Recorder:
    """Mixes the speakers (and the mic, if on) into one 16 kHz mono timeline."""

    LAG = 0.3   # seconds held back so a slower stream can still add its part

    def __init__(self, mic=False, mic_hint=""):
        self.lock = threading.Lock()
        self.rate = TARGET_RATE
        self.buf = np.zeros(TARGET_RATE * 60, np.float32)
        self.end = 0              # samples written so far (furthest point)
        self.base = 0.0           # wall time of buf[0]
        self.last_packet = 0.0
        self.running = False
        self.stopped = False
        self.sources = [Source(self, "speakers")]
        if mic:
            self.sources.append(Source(self, "mic", mic_hint))

    @property
    def error(self):
        bad = [s.kind for s in self.sources if s.error]
        return ("no " + " or ".join(bad)) if bad else ""

    def start(self):
        self.running = True
        self.base = time.time()
        for s in self.sources:
            s.thread.start()

    def stop(self):
        self.running = False
        for s in self.sources:
            s.thread.join(timeout=5)
        self.stopped = True  # nothing more is coming, so take() may have everything

    def add(self, src, a, wall_start):
        if len(a) == 0:
            return
        with self.lock:
            if self.end == 0:
                # Nothing buffered: start the timeline here (skips long silences).
                at = int(round((wall_start - self.base) * self.rate))
                if at > 0.25 * self.rate:
                    self.base += at / self.rate
                    for s in self.sources:
                        s.next = None
            at = int(round((wall_start - self.base) * self.rate))
            # Follow the stream's own sample count, unless it jumped (silence, glitch).
            if src.next is None or abs(at - src.next) > 0.25 * self.rate:
                src.next = at
            at = src.next
            src.next = at + len(a)
            if at < 0:
                a = a[-at:]
                at = 0
            stop = at + len(a)
            if stop > len(self.buf):
                bigger = np.zeros(max(stop, len(self.buf) * 2), np.float32)
                bigger[: self.end] = self.buf[: self.end]
                self.buf = bigger
            self.buf[at:stop] += a
            self.end = max(self.end, stop)
            self.last_packet = time.time()

    def _available(self):
        if self.stopped:
            return self.end
        ready = int((time.time() - self.LAG - self.base) * self.rate)
        return max(0, min(self.end, ready))

    def buffered_seconds(self):
        with self.lock:
            return self._available() / self.rate

    def peek(self):
        with self.lock:
            return self.buf[: self._available()].copy(), self.rate

    def take(self, seconds=None):
        """Remove and return (audio, wall_start, rate). All that's ready, or the first `seconds`."""
        with self.lock:
            cut = self._available()
            if seconds is not None:
                cut = min(cut, int(seconds * self.rate))
            if cut <= 0:
                return None, None, self.rate
            audio = self.buf[:cut].copy()
            start = self.base
            rest = self.end - cut
            if rest > 0:
                self.buf[:rest] = self.buf[cut:self.end]
            self.buf[max(rest, 0): self.end] = 0
            self.end = max(rest, 0)
            self.base += cut / self.rate
            for s in self.sources:
                if s.next is not None:
                    s.next -= cut
            return audio, start, self.rate


# ---------------------------------------------------------------- transcription

class Transcriber:
    """Loads whisper once and transcribes chunks from a queue, in order."""

    def __init__(self, cfg, on_event):
        self.cfg = cfg
        self.on_event = on_event      # called from this thread: (kind, payload)
        self.jobs = queue.Queue()
        self.model = None
        self.ready = threading.Event()
        self.prompt = ""
        self.carry = None        # held-back audio from the end of the last live piece
        self.carry_offset = 0.0
        threading.Thread(target=self._run, daemon=True).start()

    def _load(self):
        from faster_whisper import WhisperModel
        add_cuda_dll_dirs()
        tries = []
        if self.cfg.get("device", "") != "cpu":
            tries.append(("cuda", "float16", self.cfg.get("model") or "large-v3-turbo"))
        tries.append(("cpu", "int8", self.cfg.get("model") or "small"))
        last = None
        for device, compute, name in tries:
            try:
                self.on_event("status", f"Loading {name} model")
                model = WhisperModel(name, device=device, compute_type=compute)
                # A missing cuDNN only shows up on the first real run, so test it now.
                list(model.transcribe(np.zeros(TARGET_RATE, np.float32), beam_size=1)[0])
                log("model", name, "on", device)
                return model, device, name
            except Exception as e:
                last = e
                log("model load failed on", device, ":", e)
        raise last

    def _run(self):
        try:
            self.model, device, name = self._load()
            self.on_event("model", f"{name} ({'GPU' if device == 'cuda' else 'CPU'})")
        except Exception as e:
            self.on_event("fatal", f"Couldn't load the speech model: {e}")
            return
        self.ready.set()
        while True:
            job = self.jobs.get()
            if job[0] == "reset":
                self.prompt = ""
                self.carry = None
                continue
            if job[0] == "end":
                self.on_event("end", job[1])
                continue
            _, audio, offset, final = job
            try:
                self._transcribe(audio, offset, final)
            except Exception as e:
                log("transcribe error:", e)
            self.on_event("done_chunk", self.jobs.qsize())

    def _transcribe(self, audio, offset, final):
        if self.carry is not None:
            audio = np.concatenate([self.carry, audio])
            offset = self.carry_offset
            self.carry = None
        if len(audio) < TARGET_RATE * 0.3 or float(np.abs(audio).max()) < 0.002:
            return
        # Loopback audio is often quiet; bring it up to a steady level.
        peak = float(np.abs(audio).max())
        loud = audio * min(0.5 / peak, 20.0) if peak < 0.5 else audio
        # Earlier text helps whisper keep names and style, but text without
        # punctuation (song lyrics, a bad start) would make it drop punctuation for good.
        prompt = self.prompt[-200:]
        if not any(c in prompt for c in ".?!"):
            prompt = None
        segments, _ = self.model.transcribe(
            loud,
            language=self.cfg.get("language") or None,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            initial_prompt=prompt,
            condition_on_previous_text=False,
            word_timestamps=True,
        )
        lines = [[w for w in (seg.words or []) if w.word.strip()] for seg in segments]
        lines = [ws for ws in lines if ws]
        if not final and len(audio) < HOLD_MAX_S * TARGET_RATE:
            lines = self._hold_back(lines, audio, offset)
        for ws in lines:
            text = "".join(w.word for w in ws).strip()
            if not text:
                continue
            self.prompt = (self.prompt + " " + text)[-400:]
            self.on_event("segment", (offset + ws[0].start, text))

    def _hold_back(self, lines, audio, offset):
        """The piece was cut mid-sound, so its last words may be cut off (and whisper
        may invent an ending). Keep the last few seconds and redo them with the next
        piece, ending the kept part at a sentence end when there's one nearby."""
        words = [(i, w) for i, ws in enumerate(lines) for w in ws]
        cutoff = len(audio) / TARGET_RATE - HOLD_TAIL_S
        done = [k for k, (_, w) in enumerate(words) if w.end <= cutoff]
        n = done[-1] + 1 if done else 0
        for k in range(n - 1, -1, -1):
            w = words[k][1]
            if w.end < cutoff - 8:
                break
            if w.word.strip()[-1:] in ".?!":
                n = k + 1
                break
        if n == len(words):
            return lines
        keep_from = words[n][1].start - 0.15
        if n:
            keep_from = max(keep_from, words[n - 1][1].end)
        start = max(0, int(keep_from * TARGET_RATE))
        self.carry = audio[start:]
        self.carry_offset = offset + start / TARGET_RATE
        out = {}
        for i, w in words[:n]:
            out.setdefault(i, []).append(w)
        return [out[i] for i in sorted(out)]


# ---------------------------------------------------------------- window

class FlatButton(tk.Label):
    def __init__(self, master, text, command, primary=False, **kw):
        self.primary = primary
        self.command = command
        self.enabled = True
        super().__init__(master, text=text, cursor="hand2", **kw)
        self._paint(False)
        self.bind("<Enter>", lambda e: self._paint(True))
        self.bind("<Leave>", lambda e: self._paint(False))
        self.bind("<Button-1>", self._click)

    def _paint(self, hover):
        if self.primary:
            bg = HOVER if (hover and self.enabled) else FG
            self.configure(bg=bg if self.enabled else "#9a9a9a", fg=BG)
        else:
            self.configure(bg=BG, fg=FG if self.enabled else "#9a9a9a",
                           font=(FONT, 9, "underline" if hover and self.enabled else ""))

    def set_enabled(self, on):
        self.enabled = on
        self.configure(cursor="hand2" if on else "arrow")
        self._paint(False)

    def _click(self, _):
        if self.enabled:
            self.command()


class App:
    def __init__(self):
        self.cfg = load_config()
        self.events = queue.Queue()
        self.recorder = None
        self.state = "loading"   # loading | ready | recording | finishing | error
        self.rec_started = 0.0
        self.out_path = ""
        self.out_file = None
        self.wrote_any = False
        self.close_after = False
        self.last_saved = ""

        self.root = tk.Tk()
        self.root.title("Live Transcriber")
        self.root.configure(bg=BG)
        scale = self.root.winfo_fpixels("1i") / 96.0
        self.root.minsize(int(380 * scale), int(300 * scale))
        self.root.geometry(f"{int(480 * scale)}x{int(440 * scale)}")
        try:
            self.root.iconbitmap(ICON_PATH)
        except Exception:
            pass
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        pad = 16
        self.button = FlatButton(self.root, "Start", self.toggle, primary=True,
                                 font=(FONT, 14, "bold"), height=2)
        self.button.pack(fill="x", padx=pad, pady=(pad, 8))

        self.status = tk.Label(self.root, text="", bg=BG, fg=FG, font=(FONT, 10), anchor="w")
        self.status.pack(fill="x", padx=pad)
        self.status.bind("<Button-1>", self.open_saved)

        box = tk.Frame(self.root, bg=LINE, padx=1, pady=1)
        box.pack(fill="both", expand=True, padx=pad, pady=8)
        self.text = tk.Text(box, width=1, height=1, bg=BG, fg=FG, relief="flat", bd=0, wrap="word",
                            font=(FONT, 10), padx=8, pady=6, highlightthickness=0,
                            state="disabled", cursor="arrow")
        self.text.pack(fill="both", expand=True)
        self.text.tag_configure("time", foreground=MUTED)

        row = tk.Frame(self.root, bg=BG)
        row.pack(fill="x", padx=pad, pady=(0, pad))
        self.folder_label = tk.Label(row, text="", width=1, bg=BG, fg=MUTED, font=(FONT, 9), anchor="w")
        self.folder_label.pack(side="left", fill="x", expand=True)
        self.folder_btn = FlatButton(row, "Change folder", self.pick_folder)
        self.folder_btn.pack(side="right")
        self.mic_btn = FlatButton(row, "", self.toggle_mic)
        self.mic_btn.pack(side="right", padx=(0, 14))
        self.show_mic()
        self.show_folder()

        self.button.set_enabled(True)
        self.set_status("Loading speech model")
        self.transcriber = Transcriber(self.cfg, lambda k, p: self.events.put((k, p)))
        self.root.after(100, self.tick)

    # -- ui helpers

    def set_status(self, text, color=FG, link=False):
        self.status.configure(text=text, fg=color, cursor="hand2" if link else "arrow",
                              font=(FONT, 10, "underline" if link else ""))

    def show_folder(self):
        path = self.cfg["folder"]
        shown = path if len(path) <= 48 else "…" + path[-47:]
        self.folder_label.configure(text=shown)

    def add_line(self, t, text):
        self.text.configure(state="normal")
        self.text.insert("end", f"{fmt_time(t)}  ", "time")
        self.text.insert("end", text + "\n")
        self.text.see("end")
        self.text.configure(state="disabled")

    def pick_folder(self):
        path = filedialog.askdirectory(initialdir=self.cfg["folder"] if os.path.isdir(self.cfg["folder"]) else None,
                                       title="Save transcripts in")
        if path:
            self.cfg["folder"] = os.path.normpath(path)
            save_config(self.cfg)
            self.show_folder()

    def show_mic(self):
        self.mic_btn.configure(text="Mic on" if self.cfg.get("mic") else "Mic off")

    def toggle_mic(self):
        self.cfg["mic"] = not self.cfg.get("mic")
        save_config(self.cfg)
        self.show_mic()

    def open_saved(self, _):
        if self.last_saved and os.path.exists(self.last_saved) and self.state != "recording":
            os.startfile(self.last_saved)

    # -- recording

    def toggle(self):
        if self.state == "recording":
            self.stop()
        elif self.state in ("ready", "loading"):
            self.start()

    def start(self):
        folder = self.cfg["folder"]
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception:
            self.set_status("Can't use that folder. Pick another.", REC)
            return
        name = datetime.now().strftime("Transcript %Y-%m-%d %H-%M-%S.txt")
        self.out_path = os.path.join(folder, name)
        try:
            self.out_file = open(self.out_path, "w", encoding="utf-8")
        except Exception:
            self.set_status("Can't write to that folder. Pick another.", REC)
            return
        self.wrote_any = False
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        self.transcriber.jobs.put(("reset",))
        self.recorder = Recorder(mic=bool(self.cfg.get("mic")), mic_hint=self.cfg.get("mic_device", ""))
        self.recorder.start()
        self.rec_started = time.time()
        self.state = "recording"
        self.button.configure(text="Stop")
        self.folder_btn.set_enabled(False)
        self.mic_btn.set_enabled(False)
        self.update_rec_status()

    def stop(self):
        self.state = "finishing"
        self.recorder.stop()
        self.flush(all_audio=True)
        self.transcriber.jobs.put(("end", None))
        self.button.configure(text="Finishing")
        self.button.set_enabled(False)
        self.update_finish_status()

    def flush(self, all_audio=False):
        """Send buffered audio to whisper when it's long enough or the sound paused."""
        rec = self.recorder
        if not rec:
            return
        secs = rec.buffered_seconds()
        if secs <= 0:
            return
        if all_audio:
            self.send(*rec.take())
            return
        if secs >= MAX_CHUNK_S:
            audio, rate = rec.peek()
            self.send(*rec.take(quiet_cut(audio, rate, MAX_CHUNK_S)), final=False)
        elif time.time() - rec.last_packet > QUIET_GAP_S:
            self.send(*rec.take())
        elif secs >= MIN_CHUNK_S:
            audio, rate = rec.peek()
            tail = audio[-int(0.6 * rate):]
            if len(tail) and float(np.sqrt(np.mean(tail ** 2))) < 0.003:
                self.send(*rec.take())
            elif secs >= LIVE_CHUNK_S:
                # Sound never stops (music, background noise): cut at the quietest
                # moment of the last few seconds so text keeps showing up.
                self.send(*rec.take(quiet_cut(audio, rate, secs)), final=False)

    def send(self, audio, wall_start, rate, final=True):
        """final=False: the piece ends mid-sound, so its last sentence is held back."""
        if audio is None or len(audio) == 0:
            return
        offset = max(0.0, wall_start - self.rec_started)
        self.transcriber.jobs.put(("chunk", resample(audio, rate), offset, final))

    def update_rec_status(self):
        el = time.time() - self.rec_started
        msg = f"●  Recording  {fmt_time(el)}"
        if not self.transcriber.ready.is_set():
            msg += "   (model still loading)"
        elif self.recorder and self.recorder.error:
            msg += f"   ({self.recorder.error})"
        self.set_status(msg, REC)

    def update_finish_status(self):
        if not self.transcriber.ready.is_set():
            self.set_status("Finishing, waiting for the model to load")
        else:
            self.set_status("Finishing")

    def finish(self):
        try:
            self.out_file.close()
        except Exception:
            pass
        self.out_file = None
        self.state = "ready"
        self.button.configure(text="Start")
        self.button.set_enabled(True)
        self.folder_btn.set_enabled(True)
        self.mic_btn.set_enabled(True)
        if self.wrote_any:
            self.last_saved = self.out_path
            self.set_status("Saved  " + os.path.basename(self.out_path), FG, link=True)
        else:
            try:
                os.remove(self.out_path)
            except Exception:
                pass
            self.set_status("No speech heard, nothing saved", MUTED)
        if self.close_after:
            self.root.destroy()

    def on_close(self):
        if self.state == "recording":
            self.close_after = True
            self.stop()
        elif self.state == "finishing":
            self.close_after = True
        else:
            self.root.destroy()

    # -- loop

    def tick(self):
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "status" and self.state == "loading":
                self.set_status(payload)
            elif kind == "model":
                if self.state == "loading":
                    self.state = "ready"
                    self.set_status("Ready", MUTED)
            elif kind == "fatal":
                log(payload)
                if self.recorder:
                    self.recorder.stop()
                if self.out_file:
                    self.finish()
                self.state = "error"
                self.button.configure(text="Start")
                self.button.set_enabled(False)
                self.set_status(payload[:120], REC)
            elif kind == "segment":
                t, text = payload
                if self.out_file:
                    self.out_file.write(f"[{fmt_time(t)}] {text}\n")
                    self.out_file.flush()
                    self.wrote_any = True
                self.add_line(t, text)
            elif kind == "end":
                self.finish()
        if self.state == "recording":
            self.flush()
            self.update_rec_status()
        elif self.state == "finishing":
            self.update_finish_status()
        self.root.after(250, self.tick)

    def run(self):
        self.root.mainloop()


def quiet_cut(audio, rate, max_s):
    """Pick the quietest moment in the last 6 s before max_s, so words aren't cut in half."""
    end = int(max_s * rate)
    start = max(0, end - int(6 * rate))
    win = int(0.2 * rate)
    seg = audio[start:end]
    if len(seg) < win * 2:
        return max_s
    n = len(seg) // win
    energy = (seg[: n * win].reshape(n, win) ** 2).mean(axis=1)
    best = int(np.argmin(energy))
    return (start + best * win + win // 2) / rate


def fmt_time(s):
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def main():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    try:
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Dylan.LiveTranscriber")
    except Exception:
        pass
    App().run()


if __name__ == "__main__":
    main()
