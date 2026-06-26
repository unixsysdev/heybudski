"""
WebRTC Gateway (orchestrator) — the stateless front door.

Built on Modal's proven aiortc scaffolding (modal_webrtc.py):
  * Gateway      (ModalWebRtcSignalingServer, CPU): serves the embeddable client
                 + the /ws/{peer_id} signaling socket.
  * AvatarPeer   (ModalWebRtcPeer, CPU): one per client. Terminates WebRTC,
                 then orchestrates the GPU services purely over RPC:

      mic audio --WebRTC--> [VAD/utterance] --> ASR.remote --> LLM stream
        --> Chunker --> TTS.remote (ordered pipeline) --> audio playout track
        --> (MuseTalk frames) --> video playout track --WebRTC--> client

The Pipeline (pipeline.py) enforces the concurrency protocol: concurrent TTS
with strictly ordered playout. The peer never blocks the event loop — every
GPU call is an awaited RPC and media is moved through asyncio.Queues.

Deploy:  modal deploy gateway.py
Open the printed *-gateway-web.modal.run URL in a browser and allow the mic.

NOTE: the avatar video currently shows the static reference face; once
avatar_musetalk is deployed, _on_audio() streams real lip-synced frames
(see the TODO(musetalk) hook). The voice loop is fully functional today.
"""

import asyncio
import fractions
import json
import re
import time

import modal

from common import (
    APP_ASR,
    APP_AVATAR,
    APP_GATEWAY,
    APP_LLM,
    APP_TTS,
    ASSETS_DIR,
    assets_vol,
)
from modal_webrtc import ModalWebRtcPeer, ModalWebRtcSignalingServer

app = modal.App(APP_GATEWAY)

# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #
peer_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg")
    .pip_install(
        "aiortc==1.14.0",
        "av",
        "numpy",
        "scipy",
        "opencv-python-headless",
        "webrtcvad",
        "openai",
        "shortuuid",
        "fastapi",
    )
    .add_local_python_source("modal_webrtc", "common", "pipeline")
    .add_local_file("assets/face.jpg", "/app_assets/face.jpg")
    .add_local_file("assets/jfk.wav", "/app_assets/jfk.wav")  # test fixture
)

server_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]", "shortuuid")
    .add_local_python_source("modal_webrtc", "common")
    .add_local_dir("client", remote_path="/frontend")
)

# Handles to the deployed GPU services (resolved lazily at call time).
ASRService = modal.Cls.from_name(APP_ASR, "Whisper")
TTSService = modal.Cls.from_name(APP_TTS, "Chatterbox")
AvatarService = modal.Cls.from_name(APP_AVATAR, "Avatar")


@app.function(image=peer_image)
def health():
    """Verify the peer container can import everything + find its assets.
    Run with: modal run gateway.py::health  (no browser/WebRTC needed)."""
    import cv2  # noqa
    import aiortc  # noqa
    import av  # noqa
    import webrtcvad  # noqa
    import openai  # noqa
    import common  # noqa
    import pipeline  # noqa
    import modal_webrtc  # noqa

    img = cv2.imread("/app_assets/face.jpg")
    assert img is not None, "face.jpg missing in peer image"
    return {"ok": True, "face_shape": list(img.shape), "aiortc": aiortc.__version__}


@app.function(image=peer_image, timeout=600)
def test_brain():
    """End-to-end test of the conversational core against the REAL deployed
    ASR + LLM + TTS — exactly what AvatarPeer._respond does, minus WebRTC.
    Run: modal run gateway.py::test_brain"""
    import asyncio
    import json
    import time

    from openai import AsyncOpenAI

    from pipeline import Pipeline

    asr = modal.Cls.from_name(APP_ASR, "Whisper")()
    tts = modal.Cls.from_name(APP_TTS, "Chatterbox")()
    llm_url = modal.Function.from_name(APP_LLM, "serve").get_web_url()
    llm = AsyncOpenAI(base_url=f"{llm_url}/v1", api_key="modal", timeout=180)

    async def go():
        with open("/app_assets/jfk.wav", "rb") as f:
            wav = f.read()

        t0 = time.time()
        asr_res = await asr.transcribe.remote.aio(wav)
        user_text = asr_res.get("text", "")
        t_asr = time.time() - t0
        print(f"[test] ASR ({t_asr:.2f}s): {user_text!r}")

        history = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]
        reply, chunks = [], []
        ttff = [None]

        async def llm_stream(_):
            s = await llm.chat.completions.create(
                model="qwen", messages=history, stream=True,
                max_tokens=120, temperature=0.7,
            )
            async for ev in s:
                tok = ev.choices[0].delta.content or ""
                if tok:
                    reply.append(tok)
                    yield tok

        async def synth(c):
            return await tts.synth.remote.aio(c, VOICE)

        async def on_audio(out, c):
            if ttff[0] is None:
                ttff[0] = time.time() - t0
            chunks.append((c, round(out["samples"] / out["sr"], 2)))

        await Pipeline(llm_stream, synth, on_audio, max_inflight_tts=3).run(user_text)

        result = {
            "transcript": user_text,
            "asr_s": round(t_asr, 2),
            "reply": "".join(reply),
            "ttff_s": round(ttff[0], 2) if ttff[0] else None,
            "audio_total_s": round(sum(d for _, d in chunks), 2),
            "chunks": chunks,
        }
        print("[test] RESULT:\n" + json.dumps(result, indent=2))
        return result

    return asyncio.run(go())

VOICE = "default"  # registered on the assets Volume during TTS smoke test
PLAYOUT_SR = 48000  # WebRTC/opus clock
SYSTEM_PROMPT = (
    "You are Aria, a warm, playful, flirty female voice companion. "
    "This is a spoken conversation and you can see the whole chat history each turn. "
    "Reply in 1-2 short, natural sentences — only words you'd actually say out loud. "
    "For emotion you MAY use these exact cues in square brackets, sparingly and only "
    "when they truly fit: [laugh], [chuckle], [sigh], [gasp]. "
    "Never use any other brackets, no angle brackets, no asterisks, no emojis, no "
    "stage directions, no markdown."
)

# Chatterbox renders these paralinguistic tags as real sounds — keep them. Strip
# everything else the model might emit (other tags, <angle> actions, *stars*, emojis)
# so the voice never reads junk aloud.
_ALLOWED_TAGS = {"laugh", "chuckle", "sigh", "gasp"}
_ANGLE_STAR = re.compile(r"<[^>]*>|\*[^*]*\*")
_BRACKET = re.compile(r"\[([^\]]*)\]")
_EMOJI = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U00002B00-\U00002BFF\U0001F1E6-\U0001F1FF\U0000FE00-\U0000FE0F]"
)


def _clean_for_speech(text: str) -> str:
    t = _EMOJI.sub("", _ANGLE_STAR.sub(" ", text))
    t = _BRACKET.sub(
        lambda m: f"[{m.group(1).strip().lower()}]"
        if m.group(1).strip().lower() in _ALLOWED_TAGS
        else " ",
        t,
    )
    return " ".join(t.split())


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
def _resample(x, sr_in: int, sr_out: int):
    import numpy as np
    from math import gcd
    from scipy.signal import resample_poly

    if sr_in == sr_out:
        return x.astype(np.int16)
    g = gcd(sr_in, sr_out)
    y = resample_poly(x.astype(np.float32), sr_out // g, sr_in // g)
    return np.clip(y, -32768, 32767).astype(np.int16)


def _wav_bytes(pcm_int16, sr: int) -> bytes:
    import io
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm_int16.tobytes())
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Outgoing media tracks: pull from an asyncio.Queue, pace in real time,
# emit a fallback (silence / idle face) when the queue is empty so the
# tracks never stall.
# --------------------------------------------------------------------------- #
def _make_audio_track(queue):
    import numpy as np
    from aiortc import MediaStreamTrack
    from av import AudioFrame

    SAMPLES = PLAYOUT_SR // 50  # 20 ms

    class PlayoutAudioTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self):
            super().__init__()
            self._ts = 0
            self._start = None
            self._residual = np.zeros(0, dtype=np.int16)

        async def recv(self):
            if self._start is None:
                self._start = time.time()
            self._ts += SAMPLES
            wait = self._start + self._ts / PLAYOUT_SR - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

            buf = self._residual
            while len(buf) < SAMPLES:
                try:
                    buf = np.concatenate([buf, queue.get_nowait()])
                except asyncio.QueueEmpty:
                    break
            if len(buf) >= SAMPLES:
                out, self._residual = buf[:SAMPLES], buf[SAMPLES:]
            else:
                out = np.concatenate(
                    [buf, np.zeros(SAMPLES - len(buf), dtype=np.int16)]
                )
                self._residual = np.zeros(0, dtype=np.int16)

            frame = AudioFrame.from_ndarray(out.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = PLAYOUT_SR
            frame.pts = self._ts
            frame.time_base = fractions.Fraction(1, PLAYOUT_SR)
            return frame

    return PlayoutAudioTrack()


def _make_video_track(queue, idle_bgr, fps: int = 25):
    from aiortc import MediaStreamTrack
    from av import VideoFrame

    TICK = 90000 // fps

    class PlayoutVideoTrack(MediaStreamTrack):
        kind = "video"

        def __init__(self):
            super().__init__()
            self._ts = 0
            self._start = None
            self._last = idle_bgr

        async def recv(self):
            if self._start is None:
                self._start = time.time()
            self._ts += TICK
            wait = self._start + self._ts / 90000 - time.time()
            if wait > 0:
                await asyncio.sleep(wait)

            try:
                self._last = queue.get_nowait()
            except asyncio.QueueEmpty:
                pass  # hold last frame (idle face / last avatar frame)

            frame = VideoFrame.from_ndarray(self._last, format="bgr24")
            frame.pts = self._ts
            frame.time_base = fractions.Fraction(1, 90000)
            return frame

    return PlayoutVideoTrack()


# --------------------------------------------------------------------------- #
# Per-peer session state
# --------------------------------------------------------------------------- #
class Session:
    def __init__(self, audio_q, video_q):
        self.audio_q = audio_q
        self.video_q = video_q
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.busy = False  # don't start a new turn while one is playing
        self.dc = None      # client data channel for status/caption events
        self.avatar_id = "default"
        self.lipsync = True       # MuseTalk frames; auto-disables on error
        self.prepare_task = None  # avatar preprocessing kicked off at connect


# --------------------------------------------------------------------------- #
# The peer: terminates WebRTC, orchestrates the GPU services.
# --------------------------------------------------------------------------- #
@app.cls(
    image=peer_image,
    region="us-east",
    scaledown_window=60 * 5,
    volumes={ASSETS_DIR: assets_vol},
)
@modal.concurrent(target_inputs=2, max_inputs=4)
class AvatarPeer(ModalWebRtcPeer):
    async def initialize(self):
        import cv2
        from openai import AsyncOpenAI

        # idle avatar frame + raw bytes (the latter is sent to MuseTalk's prepare)
        with open("/app_assets/face.jpg", "rb") as f:
            self.default_face_bytes = f.read()
        img = cv2.imread("/app_assets/face.jpg")
        self.idle_bgr = cv2.resize(img, (512, 512))

        self.asr = ASRService()
        self.tts = TTSService()
        self.avatar = AvatarService()

        llm_url = modal.Function.from_name(APP_LLM, "serve").get_web_url()
        self.llm = AsyncOpenAI(base_url=f"{llm_url}/v1", api_key="modal", timeout=120)

        self.sessions: dict[str, Session] = {}
        print(f"{self.id}: initialized; llm={llm_url}")

    def _avatar_frame(self, peer_id: str):
        """Use the client's uploaded avatar (by id prefix) if present, else default."""
        import cv2

        avatar_id = peer_id.split(".")[0]
        try:
            assets_vol.reload()
            img = cv2.imread(f"{ASSETS_DIR}/avatars/{avatar_id}.img")
            if img is not None:
                print(f"{self.id}: using uploaded avatar for {avatar_id}")
                return cv2.resize(img, (512, 512))
        except Exception as e:
            print(f"{self.id}: avatar load fallback ({e})")
        return self.idle_bgr

    def _avatar_bytes(self, peer_id: str):
        """Return (image_bytes, avatar_id): uploaded avatar (by id prefix) if present."""
        import os

        avatar_id = peer_id.split(".")[0]  # peer_id = "<avatarId>.<per-connection random>"
        try:
            assets_vol.reload()
            p = f"{ASSETS_DIR}/avatars/{avatar_id}.img"
            if os.path.exists(p):
                with open(p, "rb") as f:
                    return f.read(), avatar_id
        except Exception as e:
            print(f"{self.id}: avatar bytes fallback ({e})")
        return self.default_face_bytes, "default"

    async def _prepare(self, avatar_id, img_bytes):
        """Preprocess + warm the avatar (idempotent). Returns True if lip-sync ready."""
        try:
            r = await self.avatar.prepare.remote.aio(avatar_id, img_bytes)
            print(f"{self.id}: avatar prepared {r}")
            return True
        except Exception as e:
            print(f"{self.id}: avatar prepare failed ({e}); lip-sync off")
            return False

    async def setup_streams(self, peer_id: str):
        audio_q: asyncio.Queue = asyncio.Queue()
        video_q: asyncio.Queue = asyncio.Queue()
        sess = self.sessions[peer_id] = Session(audio_q, video_q)

        img_bytes, avatar_id = self._avatar_bytes(peer_id)
        sess.avatar_id = avatar_id
        # preprocess + warm the avatar now so the first turn can lip-sync
        sess.prepare_task = asyncio.ensure_future(self._prepare(avatar_id, img_bytes))

        pc = self.pcs[peer_id]
        pc.addTrack(_make_audio_track(audio_q))
        pc.addTrack(_make_video_track(video_q, self._avatar_frame(peer_id)))

        @pc.on("connectionstatechange")
        async def _state():
            print(f"{self.id}: {peer_id} -> {pc.connectionState}")

        @pc.on("track")
        def _on_track(track):
            print(f"{self.id}: got {track.kind} track from {peer_id}")
            if track.kind == "audio":
                asyncio.ensure_future(self._consume_mic(peer_id, track))

        @pc.on("datachannel")
        def _on_dc(channel):
            self.sessions[peer_id].dc = channel
            print(f"{self.id}: data channel '{channel.label}' open from {peer_id}")

    def _emit(self, sess, **event):
        """Send a status/caption event to the browser over the data channel."""
        if sess.dc is not None:
            try:
                sess.dc.send(json.dumps(event))
            except Exception:
                pass

    # ---- microphone -> utterance detection ---- #
    async def _consume_mic(self, peer_id: str, track):
        import av
        import numpy as np
        import webrtcvad

        vad = webrtcvad.Vad(3)  # most aggressive: only clear speech passes
        # Resample whatever the browser sends to 16 kHz mono s16 — used for BOTH
        # VAD and Whisper.
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        FRAME = 320              # 20 ms @ 16 kHz
        MIN_SPEECH_FRAMES = 8    # >= ~160 ms of actual detected speech to count
        MIN_RMS = 180            # ignore near-silent audio (Whisper hallucinates on it)
        residual = np.zeros(0, dtype=np.int16)
        utter: list = []
        speaking = False
        silence_ms = 0
        speech_frames = 0

        while True:
            try:
                frame = await track.recv()
            except Exception:
                break
            out = resampler.resample(frame)
            for rf in out if isinstance(out, list) else [out]:
                if rf is None:
                    continue
                residual = np.concatenate(
                    [residual, rf.to_ndarray().reshape(-1).astype(np.int16)]
                )

            while len(residual) >= FRAME:
                f20, residual = residual[:FRAME], residual[FRAME:]
                sess = self.sessions[peer_id]
                if sess.busy:
                    # While the avatar is replying, drop mic input so we don't
                    # transcribe our own playback.
                    speaking, silence_ms, utter, speech_frames = False, 0, [], 0
                    continue
                if vad.is_speech(f20.tobytes(), 16000):
                    speaking, silence_ms = True, 0
                    speech_frames += 1
                    utter.append(f20)
                elif speaking:
                    silence_ms += 20
                    utter.append(f20)
                    if silence_ms >= 700:  # ~0.7s trailing silence => end of turn
                        audio = np.concatenate(utter)
                        n_speech = speech_frames
                        utter, speaking, silence_ms, speech_frames = [], False, 0, 0
                        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
                        if n_speech >= MIN_SPEECH_FRAMES and rms >= MIN_RMS and len(audio) >= 8000:
                            asyncio.ensure_future(self._respond(peer_id, audio))
                        else:
                            print(f"{self.id}: dropped non-speech (frames={n_speech} rms={rms:.0f})")

    # ---- the conversational turn ---- #
    async def _respond(self, peer_id: str, mic_16k):
        import cv2
        import numpy as np

        from pipeline import Pipeline

        sess = self.sessions[peer_id]
        sess.busy = True
        self._emit(sess, type="status", state="transcribing")
        try:
            wav = _wav_bytes(mic_16k, 16000)
            asr = await self.asr.transcribe.remote.aio(wav)
            user_text = (asr or {}).get("text", "").strip()
            print(f"{self.id}: user said: {user_text!r}")
            if not user_text:
                self._emit(sess, type="status", state="idle")
                return
            self._emit(sess, type="user", text=user_text)
            self._emit(sess, type="status", state="thinking")
            sess.history.append({"role": "user", "content": user_text})

            spoke = False
            reply_parts: list[str] = []

            async def llm_stream(_text):
                stream = await self.llm.chat.completions.create(
                    model="qwen",
                    messages=sess.history,
                    stream=True,
                    max_tokens=200,
                    temperature=0.7,
                )
                async for ev in stream:
                    tok = ev.choices[0].delta.content or ""
                    if tok:
                        reply_parts.append(tok)
                        yield tok

            async def synth(chunk_text):
                clean = _clean_for_speech(chunk_text)
                if not clean:
                    return None  # chunk was only a stage direction / tag
                out = await self.tts.synth.remote.aio(clean, VOICE)
                out["text"] = clean
                return out

            async def on_audio(out, _chunk):
                if out is None:
                    return
                nonlocal spoke
                self._emit(sess, type="reply", text=out["text"])
                pcm = np.frombuffer(out["pcm"], dtype=np.int16)
                audio48 = _resample(pcm, out["sr"], PLAYOUT_SR)

                # Lip-sync: render the chunk's frames FIRST and COLLECT them — do not
                # stream them out during render, or the video leads the audio. Then
                # push frames + audio together so they start aligned and the video
                # plays at a clean 25fps (not at render speed).
                frames = []
                if sess.lipsync:
                    try:
                        if sess.prepare_task is not None:
                            ok = await sess.prepare_task
                            sess.prepare_task = None
                            sess.lipsync = bool(ok)
                        if sess.lipsync:
                            a16 = _resample(pcm, out["sr"], 16000).tobytes()
                            async for jpg in self.avatar.render.remote_gen.aio(
                                sess.avatar_id, a16
                            ):
                                fr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                                if fr is not None:
                                    frames.append(cv2.resize(fr, (512, 512)))
                    except Exception as e:
                        print(f"{self.id}: lipsync error ({type(e).__name__}: {e}); disabling")
                        sess.lipsync = False

                if not spoke:
                    spoke = True
                    self._emit(sess, type="status", state="speaking")
                for fr in frames:
                    await sess.video_q.put(fr)
                step = PLAYOUT_SR // 50
                for i in range(0, len(audio48), step):
                    await sess.audio_q.put(audio48[i : i + step])

            await Pipeline(llm_stream, synth, on_audio, max_inflight_tts=3).run(user_text)
            sess.history.append({"role": "assistant", "content": "".join(reply_parts)})
            # Keep the mic muted until playback has fully drained — otherwise the
            # avatar hears its own voice (open speakers) and replies to itself.
            while not sess.audio_q.empty():
                await asyncio.sleep(0.1)
            await asyncio.sleep(0.6)
        except Exception as e:
            print(f"{self.id}: respond error: {type(e).__name__}: {e}")
            self._emit(sess, type="status", state="error")
        finally:
            self._emit(sess, type="status", state="idle")
            sess.busy = False

    async def exit(self):
        self.sessions.clear()


# --------------------------------------------------------------------------- #
# Signaling server: serves the embeddable client + WebSocket signaling.
# --------------------------------------------------------------------------- #
@app.cls(image=server_image, volumes={ASSETS_DIR: assets_vol})
class Gateway(ModalWebRtcSignalingServer):
    def get_modal_peer_class(self):
        return AvatarPeer

    def initialize(self):
        import os

        from fastapi import File, UploadFile
        from fastapi.responses import HTMLResponse
        from fastapi.staticfiles import StaticFiles

        self.web_app.mount("/static", StaticFiles(directory="/frontend"))

        @self.web_app.get("/")
        async def root():
            return HTMLResponse(open("/frontend/index.html").read())

        @self.web_app.post("/upload_avatar/{peer_id}")
        async def upload_avatar(peer_id: str, file: UploadFile = File(...)):
            data = await file.read()
            if len(data) > 12_000_000:
                return {"ok": False, "error": "image too large (max 12MB)"}
            pid = "".join(c for c in peer_id if c.isalnum() or c in "-_")[:40]
            os.makedirs(f"{ASSETS_DIR}/avatars", exist_ok=True)
            with open(f"{ASSETS_DIR}/avatars/{pid}.img", "wb") as f:
                f.write(data)
            assets_vol.commit()
            print(f"[gateway] stored avatar for {pid} ({len(data)} bytes)")
            return {"ok": True, "peer_id": pid, "bytes": len(data)}
