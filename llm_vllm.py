"""
Language engine — Qwen3.6-35B-A3B (35B total / 3B active MoE) via vLLM's
OpenAI-compatible server on an A100-80GB.

Budget-critical detail: weights (~70GB) are pulled to the Volume by a CPU
function (download_model) FIRST, so the A100 only ever pays for inference,
never for the download. Boot the GPU only after download_model has run once.

Usage:
  modal run llm_vllm.py::download_model      # one-time, cheap CPU, ~70GB
  modal deploy llm_vllm.py                    # brings up the A100 vLLM server
  # then: curl https://<workspace>--ai-buddy-llm-serve.modal.run/v1/models
"""

import subprocess

import modal

from common import APP_LLM, GPU_LLM, QWEN_MODEL_ID, HF_CACHE_DIR, hf_cache_vol

app = modal.App(APP_LLM)

MINUTES = 60

# vLLM's PyPI wheel bundles CUDA. hf_transfer for fast 70GB pulls.
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm>=0.19.0", "huggingface_hub[hf_transfer]")
    .env(
        {
            "HF_HOME": HF_CACHE_DIR,
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "VLLM_USE_V1": "1",
            # FlashInfer JIT-compiles sampling kernels and needs nvcc, which the
            # torch-runtime image lacks. Use vLLM's native sampler instead.
            "VLLM_USE_FLASHINFER_SAMPLER": "0",
        }
    )
    .add_local_python_source("common")
)


@app.function(
    image=vllm_image,
    volumes={HF_CACHE_DIR: hf_cache_vol},
    timeout=60 * MINUTES,
)
def download_model():
    """One-time: pull weights to the Volume on cheap CPU (no GPU billed)."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        QWEN_MODEL_ID,
        ignore_patterns=["*.pt", "*.bin", "*.pth"],  # prefer safetensors
    )
    hf_cache_vol.commit()
    print(f"[llm] downloaded {QWEN_MODEL_ID} -> {path}")


@app.function(
    image=vllm_image,
    gpu=GPU_LLM,
    volumes={HF_CACHE_DIR: hf_cache_vol},
    scaledown_window=10 * MINUTES,  # stay warm 10 min after last token, then to zero
    timeout=60 * MINUTES,
)
@modal.concurrent(max_inputs=64)  # one server handles many concurrent HTTP streams
@modal.web_server(port=8000, startup_timeout=30 * MINUTES)
def serve():
    """vLLM OpenAI-compatible server. Gateway streams chat completions from it."""
    cmd = (
        f"vllm serve {QWEN_MODEL_ID} "
        "--host 0.0.0.0 --port 8000 "
        "--served-model-name qwen "
        "--max-model-len 8192 "          # ample conversational context
        "--gpu-memory-utilization 0.90 "
        "--max-num-seqs 16 "             # plenty of concurrent turns for this app; fits L4
        "--enable-prefix-caching "        # reuse system-prompt/history KV across turns
        # Skip torch.compile/CUDA-graph capture -> ~60s cold boot, reliable.
        # A 7B in eager mode is still well within real-time.
        "--enforce-eager"
    )
    print(f"[llm] launching: {cmd}")
    subprocess.Popen(cmd, shell=True)


@app.local_entrypoint()
def smoke():
    """Quick sanity check against the deployed server (run after `modal deploy`)."""
    import urllib.request
    import json

    url = serve.get_web_url()
    print(f"[llm] server url: {url}")
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(
            {
                "model": "qwen",
                "messages": [{"role": "user", "content": "Say hi in 5 words."}],
                "max_tokens": 32,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        print("[llm] response:", json.loads(r.read())["choices"][0]["message"]["content"])
