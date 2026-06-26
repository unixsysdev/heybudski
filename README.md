# AI Buddy — Real-Time Interactive AI Avatar

> 🔴 **Live demo: [heybudski.com](https://heybudski.com)**

An embeddable, real-time conversational AI avatar. Open the page, hit **Start call**,
and talk to an on-screen face that listens, thinks, and replies out loud — with its
**mouth lip-synced to the speech**, streaming, with no "press to send".

It runs as a set of serverless GPU microservices on [Modal](https://modal.com): each
model is its own auto-scaling service, tied together by a stateless WebRTC gateway.

---

## How it works

```
 Browser ──WebRTC(mic)──▶  Gateway (CPU, aiortc)
                              │  1. VAD + energy gate → find end of your turn
                              │  2. transcribe            ──▶  ASR    (faster-whisper, GPU)
                              │  3. stream a reply         ──▶  LLM    (Qwen3-4B / vLLM, GPU)
                              │  4. chunk (sentence / tag / long-enough comma)
                              │  5. synthesize each chunk  ──▶  TTS    (Chatterbox Turbo, GPU)
                              │  6. drive the mouth         ──▶  Avatar (MuseTalk, GPU)
 Browser ◀─WebRTC(audio+video)─┘  7. play audio + lip-synced video, with live captions
```

It's an **asynchronous streaming loop** — the system never waits for a full sentence
or a complete audio file before starting the next step.

### A single turn, step by step

1. **Listen.** The browser streams microphone audio over WebRTC. The gateway resamples
   it to 16 kHz and uses voice-activity detection plus an energy gate to find where your
   turn ends — and to ignore silence and room noise (so the model never "hears" things
   you didn't say).
2. **Transcribe.** The captured utterance → speech-to-text → text.
3. **Think (streaming).** The text goes to the language model, which streams tokens back
   as it generates — downstream work starts before it's finished.
4. **Chunk.** A small buffer splits the token stream at sentence ends, emotion tags, or
   commas — but a comma only cuts once the clause already has enough words, so speech
   starts quickly without choppy one-word fragments.
5. **Speak.** Each chunk → the voice engine, which renders inline emotion tags
   (`[laugh]`, `[sigh]`, …) as real vocal sounds.
6. **Animate.** The synthesized audio drives the avatar's mouth, producing lip-synced
   video frames.
7. **Stream back.** Audio and video go to the browser over WebRTC, time-aligned, with
   live captions of what you said and what the avatar is saying.

### The concurrency model

Synthesizing every chunk concurrently and emitting whichever finishes first would
**garble speech** (a short later sentence can finish before a long earlier one). The
pipeline keeps the speed of concurrency without the disorder:

- TTS for upcoming chunks runs ahead of playback (latency hiding).
- A single consumer **awaits the chunks in submission order**, so audio and lips are
  always in order.
- A bounded number of in-flight TTS jobs provides back-pressure all the way up to LLM
  token consumption.
- For each chunk, the avatar frames are collected and pushed **together** with the
  audio, so the two stay in sync.

This logic lives in `pipeline.py` and has a self-test that runs with no GPU:

```bash
python3 pipeline.py
```

---

## Components

| Service | Role | Model |
| --- | --- | --- |
| `asr_whisper.py` | speech → text | `faster-whisper` (distil-large-v3) |
| `llm_vllm.py` | conversation | **Qwen3-4B-Instruct-2507** via `vLLM` (non-thinking) |
| `tts_chatterbox.py` | text → speech, voice cloning, emotion tags | Resemble AI **Chatterbox Turbo** |
| `avatar_musetalk.py` | audio-driven lip-sync video | **MuseTalk** (latent-space inpainting) |
| `gateway.py` | WebRTC, orchestration, the streaming pipeline | FastAPI + `aiortc` |
| `pipeline.py` | the ordered streaming pipeline + chunker | — |
| `common.py` | shared config (model ids, GPUs, volumes) | — |
| `client/` | embeddable browser front-end (mic in, lip-synced avatar out, avatar upload) | — |

### Why these models

- **Voice — Chatterbox Turbo:** small and fast, zero-shot voice cloning from a short
  reference clip, and it renders inline emotion tags as real vocal sounds.
- **Avatar — MuseTalk:** repaints only the mouth region instead of regenerating the
  whole frame, which is what makes real-time frame rates possible.
- **Brain — Qwen3-4B-Instruct-2507:** recent and capable, and **non-thinking** — it
  never burns time on hidden reasoning, so replies start fast. Served through vLLM's
  streaming API. The active model is a one-line change in `common.py`.

---

## Personalize the avatar

The front-end has a **"Use my own avatar"** button — upload a face and it's
face-detected, prepared, and lip-synced for your session. A stable per-browser id keeps
your choice across page refreshes. The default face and voice are configured in the repo
(`assets/`), and the conversational persona is set in the gateway's system prompt.

---

## Deploy

Requires the Modal CLI, installed and authenticated. Each service is deployed
independently and scales to zero when idle.

```bash
# 1. One-time: pull model weights onto Volumes (runs on CPU, cached)
modal run llm_vllm.py::download_model
modal run avatar_musetalk.py::download_weights

# 2. Register the default cloned voice
modal run tts_chatterbox.py

# 3. Deploy every service
for f in asr_whisper tts_chatterbox llm_vllm avatar_musetalk gateway; do
  modal deploy $f.py
done
```

`modal deploy gateway.py` prints the public URL — open it, allow the microphone, and
start talking.

### Smoke-test pieces individually (no browser)

```bash
modal run asr_whisper.py            # transcribe a sample clip
modal run tts_chatterbox.py         # synthesize a line (with emotion tags)
modal run avatar_musetalk.py        # prepare a face + render lip-synced frames
modal run gateway.py::test_brain    # full ASR → LLM → TTS round-trip + timing
```

---

## Project layout

```
common.py            shared config (model ids, GPUs, volume names)
pipeline.py          ordered streaming pipeline + chunker (self-testable)
asr_whisper.py       speech-to-text service
tts_chatterbox.py    text-to-speech + voice cloning service
llm_vllm.py          language-model service (vLLM streaming server)
avatar_musetalk.py   MuseTalk lip-sync avatar service
gateway.py           WebRTC gateway + orchestrator
modal_webrtc.py      WebRTC-on-Modal signaling helper
client/              embeddable browser front-end
assets/              default face + voice reference
```

`modal_webrtc.py` is adapted from the
[modal-examples WebRTC sample](https://github.com/modal-labs/modal-examples/tree/main/07_web/webrtc).

---

## Planned features

Not implemented yet — on the roadmap:

- **Voice cloning from the UI** — upload a short voice sample in the browser to clone a
  custom voice for the avatar. (The backend already supports zero-shot cloning; this adds
  the in-app upload and per-session voice selection.)
- **Personality upload** — set the avatar's persona by uploading a Markdown (`.md`) file
  that becomes the system prompt, so you can fully customize how it talks and behaves.

---

## Notes

- **Cold starts:** services scale to zero, so the first request after idle pays a
  model-load cost. Keep a service warm during a session for the snappiest first reply.
- **Echo:** use headphones — open speakers can feed the avatar's voice back into the
  mic. Browser echo-cancellation is on, and the gateway mutes the mic while the avatar
  is speaking.
