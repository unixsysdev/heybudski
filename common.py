"""
Shared configuration for the AI.Buddy real-time interactive avatar stack on Modal.

Architecture: each GPU service is its own Modal *app* so it scales to zero
independently. The CPU gateway (ai-buddy-gateway) looks them up by name with
modal.Cls.from_name / modal.Function.from_name and calls them over RPC.

Cost note (verified Modal pricing, 2026): A100-80GB ~$2.50/hr, L4 ~$0.80/hr.
All-warm (LLM + TTS + ASR + avatar) ~= $4-5/hr, so $30 ~= 6-7h of live runtime.
Everything below is scale-to-zero; only the gateway warms the GPUs per session.
"""

import modal

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
# Active model. Qwen3-4B-Instruct-2507: recent Qwen3 series, 4B (fast + cheap on an
# L4), non-thinking (no <think> blocks), 262K context, Apache-2.0 — great fit for a
# snappy real-time voice companion.
#
# The spec's Qwen3.6-35B-A3B (hybrid Gated-DeltaNet MoE) is one line away, but on
# current vLLM it needs an A100-80GB, ~5-7min cold starts, and hits several
# bleeding-edge-arch issues (FlashInfer/nvcc JIT, Mamba-cache sizing, untuned MoE).
# Swap back here (and set GPU_LLM="A100-80GB") once that support matures.
QWEN_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

# Resemble AI Chatterbox Turbo: 350M, ~75ms, native paralinguistic tags, MIT.
CHATTERBOX_MODEL_ID = "ResembleAI/chatterbox-turbo"

# faster-whisper / CTranslate2 model id. "distil-large-v3" is fast+accurate;
# swap to "small" or "base" to cut ASR cost during dev.
WHISPER_MODEL_ID = "distil-large-v3"

# MuseTalk 1.5 (latent-space inpainting lip-sync).
MUSETALK_REPO = "TMElyralab/MuseTalk"

# --------------------------------------------------------------------------- #
# App names (one app per service for independent lifecycle / scale-to-zero)
# --------------------------------------------------------------------------- #
APP_LLM = "ai-buddy-llm"
APP_TTS = "ai-buddy-tts"
APP_ASR = "ai-buddy-asr"
APP_AVATAR = "ai-buddy-avatar"
APP_GATEWAY = "ai-buddy-gateway"

# --------------------------------------------------------------------------- #
# GPUs
# --------------------------------------------------------------------------- #
GPU_LLM = "L4"  # Qwen2.5-7B fits comfortably (~15GB) and is cheap; use "A100-80GB" for the 35B
GPU_TTS = "L4"
GPU_ASR = "L4"
GPU_AVATAR = "L4"

# --------------------------------------------------------------------------- #
# Shared volumes
# --------------------------------------------------------------------------- #
# Big model-weight cache (HF hub), shared read-mostly across services.
hf_cache_vol = modal.Volume.from_name("ai-buddy-hf-cache", create_if_missing=True)
# Per-user derived assets: cloned-voice embeddings, MuseTalk face masks/latents.
assets_vol = modal.Volume.from_name("ai-buddy-assets", create_if_missing=True)

HF_CACHE_DIR = "/cache"
ASSETS_DIR = "/assets"

# --------------------------------------------------------------------------- #
# Audio constants
# --------------------------------------------------------------------------- #
ASR_SR = 16000       # whisper expects 16 kHz mono
MUSETALK_SR = 16000  # MuseTalk's Whisper audio-feature encoder expects 16 kHz
# Chatterbox output sample rate (~24 kHz) is read from the model at runtime and
# resampled to MUSETALK_SR before being fed to the avatar.
