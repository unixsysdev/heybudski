"""
ASR service — faster-whisper (CTranslate2) on an L4.

Transcribes audio bytes -> text. The gateway streams the client's mic audio
here. Scales to zero between sessions.

Smoke test:   modal run asr_whisper.py
Deploy:       modal deploy asr_whisper.py
"""

import modal

from common import (
    APP_ASR,
    GPU_ASR,
    WHISPER_MODEL_ID,
    HF_CACHE_DIR,
    ASR_SR,
    hf_cache_vol,
)

app = modal.App(APP_ASR)

# CUDA 12.4 + cuDNN 9 runtime — matches CTranslate2 4.5+ (faster-whisper's backend).
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04", add_python="3.11"
    )
    # huggingface_hub 1.x dropped the transitive `requests` dep that
    # faster-whisper 1.1.1 still imports directly — add it explicitly.
    .pip_install("faster-whisper==1.1.1", "requests")
    .env({"HF_HOME": HF_CACHE_DIR})
    .add_local_python_source("common")
)


@app.cls(
    gpu=GPU_ASR,
    image=image,
    volumes={HF_CACHE_DIR: hf_cache_vol},
    scaledown_window=120,   # stay warm 2 min after last call, then scale to zero
    min_containers=0,
)
class Whisper:
    @modal.enter()
    def load(self):
        from faster_whisper import WhisperModel

        # float16 on L4; int8_float16 is faster but slightly less accurate.
        self.model = WhisperModel(
            WHISPER_MODEL_ID, device="cuda", compute_type="float16"
        )
        print(f"[asr] loaded {WHISPER_MODEL_ID}")

    @modal.method()
    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> dict:
        """Decode arbitrary audio bytes (any container/SR), resample to 16k, transcribe."""
        import io
        from faster_whisper.audio import decode_audio

        # PyAV-backed decode + resample to mono 16k float32.
        audio = decode_audio(io.BytesIO(audio_bytes), sampling_rate=ASR_SR)

        segments, info = self.model.transcribe(
            audio,
            language=language,
            beam_size=1,          # greedy = lowest latency
            vad_filter=True,      # drop non-speech so Whisper doesn't hallucinate
                                  # "Thank you." on silence
            no_speech_threshold=0.6,
            condition_on_previous_text=False,
        )
        text = "".join(seg.text for seg in segments).strip()
        return {
            "text": text,
            "language": info.language,
            "duration": round(info.duration, 2),
        }


@app.local_entrypoint()
def main(audio_path: str = "assets/jfk.wav"):
    with open(audio_path, "rb") as f:
        data = f.read()
    print(f"[asr] sending {len(data)} bytes from {audio_path} ...")
    result = Whisper().transcribe.remote(data)
    print("[asr] result:", result)
