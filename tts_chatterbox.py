"""
Voice service — Chatterbox Turbo (Resemble AI) on an L4.

Two-phase per the spec:
  * register_voice(name, audio_bytes): one-time — store the 5-10s reference clip
    on the assets Volume (zero-shot clone source). Conditionals are then prepared
    once per voice and reused, so per-chunk synth stays in the ~75ms regime.
  * synth(text, voice): runtime — text chunk (with inline paralinguistic tags
    like [laugh]) -> raw int16 PCM bytes + sample rate.

Smoke test:   modal run tts_chatterbox.py
Deploy:       modal deploy tts_chatterbox.py
"""

import modal

from common import (
    APP_TTS,
    GPU_TTS,
    HF_CACHE_DIR,
    ASSETS_DIR,
    hf_cache_vol,
    assets_vol,
)

app = modal.App(APP_TTS)

# torch's PyPI wheel bundles its own CUDA runtime, so debian_slim is enough.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install("chatterbox-tts", "numpy", "soundfile")
    .env({"HF_HOME": HF_CACHE_DIR})
    .add_local_python_source("common")
)

VOICES_DIR = f"{ASSETS_DIR}/voices"


@app.cls(
    gpu=GPU_TTS,
    image=image,
    volumes={HF_CACHE_DIR: hf_cache_vol, ASSETS_DIR: assets_vol},
    scaledown_window=120,
    min_containers=0,
)
class Chatterbox:
    @modal.enter()
    def load(self):
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        self.torch = torch
        self._current_voice = None
        self._ref_path = None
        self._prepared = False
        print(f"[tts] loaded Chatterbox Turbo, sr={self.model.sr}")

    def _ensure_voice(self, voice_name: str):
        """Prepare (and cache) speaker conditionals for a voice, once."""
        if self._current_voice == voice_name:
            return
        path = f"{VOICES_DIR}/{voice_name}.wav"
        self._ref_path = path
        self._prepared = False
        # Fast path: precompute conditionals once so generate() skips re-extraction.
        if hasattr(self.model, "prepare_conditionals"):
            try:
                self.model.prepare_conditionals(path)
                self._prepared = True
            except Exception as e:  # fall back to per-call audio_prompt_path
                print(f"[tts] prepare_conditionals fallback: {e}")
        self._current_voice = voice_name

    @modal.method()
    def register_voice(self, name: str, audio_bytes: bytes) -> dict:
        """One-time: persist a reference clip to the assets Volume for cloning."""
        import os

        os.makedirs(VOICES_DIR, exist_ok=True)
        path = f"{VOICES_DIR}/{name}.wav"
        with open(path, "wb") as f:
            f.write(audio_bytes)
        assets_vol.commit()  # make visible to other containers immediately
        print(f"[tts] registered voice '{name}' ({len(audio_bytes)} bytes)")
        return {"voice": name, "path": path, "bytes": len(audio_bytes)}

    @modal.method()
    def synth(self, text: str, voice: str) -> dict:
        """Text chunk -> int16 PCM bytes at model.sr. Paralinguistic tags inline."""
        import numpy as np

        self._ensure_voice(voice)
        with self.torch.inference_mode():
            if self._prepared:
                wav = self.model.generate(text)
            else:
                wav = self.model.generate(text, audio_prompt_path=self._ref_path)

        arr = wav.squeeze().detach().cpu().numpy().astype("float32")
        pcm16 = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        return {"pcm": pcm16, "sr": int(self.model.sr), "samples": int(arr.shape[-1])}


@app.local_entrypoint()
def main(voice_ref: str = "assets/female.wav"):
    with open(voice_ref, "rb") as f:
        ref = f.read()

    box = Chatterbox()
    print("[tts] registering voice ...")
    print("  ", box.register_voice.remote("default", ref))

    text = "Hey there [chuckle], it's so good to finally hear from you. [sigh] I missed this."
    print(f"[tts] synthesizing: {text!r}")
    out = box.synth.remote(text, "default")
    dur = out["samples"] / out["sr"]
    print(f"[tts] got {len(out['pcm'])} PCM bytes @ {out['sr']} Hz = {dur:.2f}s audio")
