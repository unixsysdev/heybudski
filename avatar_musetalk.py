"""
Avatar service — MuseTalk (audio-driven lip-sync) on a GPU.

Two-phase, mirroring the spec:
  * prepare(avatar_id, image_bytes): one-time — run face detection + VAE latents +
    face-parsing masks on a reference image, cache them to the assets Volume.
  * render(avatar_id, audio_bytes): runtime — generator that takes a chunk of 16kHz
    audio and yields lip-synced BGR video frames (JPEG-encoded for cheap transport).

This reimplements the essential logic of MuseTalk's scripts/realtime_inference.py
(the Avatar class) without its file-I/O, interactive prompts, or module globals, so
it can be driven as a Modal generator from the gateway.

  modal run avatar_musetalk.py::download_weights   # one-time, CPU, ~several GB
  modal run avatar_musetalk.py                      # smoke test (prepare + render)
  modal deploy avatar_musetalk.py
"""

import modal

from common import (
    APP_AVATAR,
    GPU_AVATAR,
    HF_CACHE_DIR,
    ASSETS_DIR,
    MUSETALK_SR,
    hf_cache_vol,
    assets_vol,
)

app = modal.App(APP_AVATAR)

MT_DIR = "/MuseTalk"
MODELS_DIR = f"{MT_DIR}/models"

# MuseTalk targets torch 2.0.1 / cu118 + the OpenMMLab vision stack. Use a CUDA
# devel base so mmcv can compile if a prebuilt wheel isn't available.
avatar_image = (
    modal.Image.from_registry(
        "nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04", add_python="3.10"
    )
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0", "build-essential", "wget")
    .run_commands(f"git clone https://github.com/TMElyralab/MuseTalk.git {MT_DIR}")
    .pip_install(
        "torch==2.0.1",
        "torchvision==0.15.2",
        "torchaudio==2.0.2",
        index_url="https://download.pytorch.org/whl/cu118",
    )
    .run_commands(f"cd {MT_DIR} && pip install -r requirements.txt")
    .pip_install("openmim")
    .run_commands(
        "mim install mmengine",
        "mim install 'mmcv==2.0.1'",
        "mim install 'mmdet==3.1.0'",
        # chumpy (an mmpose dep) ships an ancient setup.py that `import pip`s and
        # breaks under PEP 517 build isolation — install it without isolation first
        # so mmpose finds it already satisfied.
        "pip install --no-build-isolation chumpy",
        "mim install 'mmpose==1.1.0'",
    )
    .pip_install("gdown", "huggingface_hub")
    # torch 2.0.1 / torchvision 0.15.2 are built against numpy 1.x — a stray numpy 2.x
    # (pulled in by another dep) breaks their C extensions with the pybind ABI error
    # "Unable to convert function return value ... () -> handle". Pin numpy<2 last.
    .run_commands("pip install 'numpy<2.0'")
    .env({"HF_HOME": HF_CACHE_DIR, "PYTHONPATH": MT_DIR})
    .add_local_python_source("common")
)

# Weights live on the HF-cache Volume (mounted into MuseTalk's models/ dir).
weights_vol = hf_cache_vol
AVATARS_SUBDIR = "musetalk_avatars"  # prepared latents/coords/masks on assets_vol


@app.function(image=avatar_image, volumes={MODELS_DIR: weights_vol}, timeout=60 * 30)
def download_weights():
    """Download MuseTalk's inference weights to the models Volume (skip syncnet)."""
    import os

    # hf_transfer's xet CDN 403s on these unauthenticated public files — use the
    # stable download path (must be set before huggingface_hub is imported).
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    import subprocess

    from huggingface_hub import snapshot_download

    def hf(repo, patterns, sub=""):
        snapshot_download(
            repo, local_dir=os.path.join(MODELS_DIR, sub) if sub else MODELS_DIR,
            allow_patterns=patterns,
        )

    hf("TMElyralab/MuseTalk", ["musetalk/*", "musetalkV15/*"])
    hf("stabilityai/sd-vae-ft-mse", ["config.json", "diffusion_pytorch_model.bin"], "sd-vae")
    hf("openai/whisper-tiny", ["config.json", "pytorch_model.bin", "preprocessor_config.json"], "whisper")
    hf("yzd-v/DWPose", ["dw-ll_ucoco_384.pth"], "dwpose")

    # Face-parse weights: BiSeNet from Google Drive + resnet18 from pytorch.
    fp = os.path.join(MODELS_DIR, "face-parse-bisent")
    os.makedirs(fp, exist_ok=True)
    bisenet = os.path.join(fp, "79999_iter.pth")
    # gdown 6.x dropped --id; pass the id positionally. Fall back to an HF mirror.
    subprocess.run(["gdown", "154JgKpzCPW82qINcVieuPH3fZ2e0P812", "-O", bisenet], check=False)
    if not os.path.exists(bisenet) or os.path.getsize(bisenet) < 1_000_000:
        try:
            from huggingface_hub import hf_hub_download

            src = hf_hub_download("camenduru/MuseTalk", "models/face-parse-bisent/79999_iter.pth")
            subprocess.run(["cp", src, bisenet], check=False)
        except Exception as e:
            print(f"[avatar] bisenet HF fallback failed: {e}")
    subprocess.run(
        ["wget", "-q", "https://download.pytorch.org/models/resnet18-5c106cde.pth",
         "-O", os.path.join(fp, "resnet18-5c106cde.pth")], check=False)

    weights_vol.commit()
    print("[avatar] weights downloaded:", os.listdir(MODELS_DIR))
    print("[avatar] face-parse-bisent:", os.listdir(fp))


@app.cls(
    image=avatar_image,
    gpu=GPU_AVATAR,
    volumes={MODELS_DIR: weights_vol, ASSETS_DIR: assets_vol},
    scaledown_window=60 * 5,
    timeout=60 * 30,
)
class Avatar:
    @modal.enter()
    def load(self):
        import os
        import sys

        import torch

        os.chdir(MT_DIR)  # MuseTalk resolves "models/..." relative to the repo root
        sys.path.insert(0, MT_DIR)
        from musetalk.utils.utils import load_all_model
        from musetalk.utils.audio_processor import AudioProcessor
        from musetalk.utils.face_parsing import FaceParsing
        from transformers import WhisperModel

        self.torch = torch
        self.device = torch.device("cuda")

        # v1.5 weights
        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path=f"{MODELS_DIR}/musetalkV15/unet.pth",
            vae_type="sd-vae",
            unet_config=f"{MODELS_DIR}/musetalkV15/musetalk.json",
            device=self.device,
        )
        self.timesteps = torch.tensor([0], device=self.device)
        self.pe = self.pe.half().to(self.device)
        self.vae.vae = self.vae.vae.half().to(self.device)
        self.unet.model = self.unet.model.half().to(self.device)
        self.weight_dtype = self.unet.model.dtype

        self.audio_processor = AudioProcessor(feature_extractor_path=f"{MODELS_DIR}/whisper")
        self.whisper = (
            WhisperModel.from_pretrained(f"{MODELS_DIR}/whisper")
            .to(device=self.device, dtype=self.weight_dtype)
            .eval()
        )
        self.whisper.requires_grad_(False)
        self.fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)
        self._cache = {}  # avatar_id -> dict(latents, coords, frames, masks, mask_boxes)
        print("[avatar] models loaded")

    # ------------------------------------------------------------------ #
    def _avatar_dir(self, avatar_id):
        return f"{ASSETS_DIR}/{AVATARS_SUBDIR}/{avatar_id}"

    @modal.method()
    def prepare(self, avatar_id: str, image_bytes: bytes, force: bool = False) -> dict:
        """Face detect + VAE latents + parsing masks for one reference image; cache it.
        Idempotent: returns immediately if this avatar was already prepared."""
        import os
        import pickle

        import cv2
        import numpy as np

        from musetalk.utils.preprocessing import get_landmark_and_bbox
        from musetalk.utils.blending import get_image_prepare_material

        d = self._avatar_dir(avatar_id)
        if not force:
            try:
                assets_vol.reload()
            except Exception:
                pass
            if os.path.exists(f"{d}/latents.pt") and os.path.exists(f"{d}/meta.pkl"):
                return {"avatar_id": avatar_id, "cached": True}
        os.makedirs(d, exist_ok=True)
        img_path = f"{d}/face.png"
        arr = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        cv2.imwrite(img_path, arr)

        coords, frames = get_landmark_and_bbox([img_path], 0)
        latents = []
        for bbox, frame in zip(coords, frames):
            x1, y1, x2, y2 = bbox
            y2 = min(y2 + 10, frame.shape[0])  # v1.5 extra_margin
            crop = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents.append(self.vae.get_latents_for_unet(crop))

        masks, mask_boxes = [], []
        for bbox, frame in zip(coords, frames):
            x1, y1, x2, y2 = bbox
            y2 = min(y2 + 10, frame.shape[0])
            mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=self.fp, mode="jaw")
            masks.append(mask)
            mask_boxes.append(crop_box)

        self.torch.save(latents, f"{d}/latents.pt")
        with open(f"{d}/meta.pkl", "wb") as f:
            pickle.dump({"coords": coords, "frames": frames, "masks": masks, "mask_boxes": mask_boxes}, f)
        assets_vol.commit()
        self._cache.pop(avatar_id, None)
        print(f"[avatar] prepared '{avatar_id}' ({len(latents)} ref frame(s))")
        return {"avatar_id": avatar_id, "frames": len(latents)}

    def _load_avatar(self, avatar_id):
        import pickle

        if avatar_id in self._cache:
            return self._cache[avatar_id]
        d = self._avatar_dir(avatar_id)
        latents = self.torch.load(f"{d}/latents.pt")
        with open(f"{d}/meta.pkl", "rb") as f:
            meta = pickle.load(f)
        data = {"latents": latents, **meta}
        self._cache[avatar_id] = data
        return data

    @modal.method()
    def render(self, avatar_id: str, audio_bytes: bytes, fps: int = 25):
        """Generator: 16kHz audio chunk -> JPEG-encoded lip-synced BGR frames."""
        import copy
        import os
        import tempfile

        import cv2
        import numpy as np

        from musetalk.utils.utils import datagen
        from musetalk.utils.blending import get_image_blending

        a = self._load_avatar(avatar_id)
        latent_cycle = a["latents"] + a["latents"][::-1]
        coord_cycle = a["coords"] + a["coords"][::-1]
        frame_cycle = a["frames"] + a["frames"][::-1]
        mask_cycle = a["masks"] + a["masks"][::-1]
        box_cycle = a["mask_boxes"] + a["mask_boxes"][::-1]

        # Whisper features expect a file; write the chunk to a temp wav.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            import wave

            with wave.open(tf.name, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(MUSETALK_SR)
                w.writeframes(audio_bytes)
            wav_path = tf.name

        feats, length = self.audio_processor.get_audio_feature(wav_path, weight_dtype=self.weight_dtype)
        chunks = self.audio_processor.get_whisper_chunk(
            feats, self.device, self.weight_dtype, self.whisper, length,
            fps=fps, audio_padding_length_left=2, audio_padding_length_right=2,
        )
        os.unlink(wav_path)

        idx = 0
        for whisper_batch, latent_batch in datagen(chunks, latent_cycle, batch_size=4):
            audio_feat = self.pe(whisper_batch.to(self.device))
            latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)
            pred = self.unet.model(latent_batch, self.timesteps, encoder_hidden_states=audio_feat).sample
            pred = pred.to(device=self.device, dtype=self.vae.vae.dtype)
            for res in self.vae.decode_latents(pred):
                n = len(coord_cycle)
                bbox = coord_cycle[idx % n]
                ori = copy.deepcopy(frame_cycle[idx % n])
                x1, y1, x2, y2 = bbox
                y2 = min(y2 + 10, ori.shape[0])
                try:
                    res_frame = cv2.resize(res.astype(np.uint8), (x2 - x1, y2 - y1))
                except Exception:
                    idx += 1
                    continue
                combined = get_image_blending(
                    ori, res_frame, [x1, y1, x2, y2], mask_cycle[idx % n], box_cycle[idx % n]
                )
                ok, buf = cv2.imencode(".jpg", combined, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    yield buf.tobytes()
                idx += 1


@app.local_entrypoint()
def main(face: str = "assets/face.jpg", audio: str = "assets/female.wav", force: bool = True):
    import wave

    with open(face, "rb") as f:
        img = f.read()
    av = Avatar()
    print(av.prepare.remote("default", img, force))

    # 16k mono PCM from the sample wav
    with wave.open(audio, "rb") as w:
        pcm = w.readframes(w.getnframes())
    n = sum(1 for _ in av.render.remote_gen("default", pcm))
    print(f"[avatar] rendered {n} frames")
