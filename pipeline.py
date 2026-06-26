"""
The critical concurrency protocol — ordered streaming pipeline.

    LLM tokens -> Chunker -> concurrent TTS (bounded) -> STRICTLY ORDERED audio
    -> single consumer (drives the avatar) -> frames

Why this shape (and what it fixes in the spec's pseudo-code):

  * The spec did `tts_task.add_done_callback(lambda t: pipe_to_avatar(...))`.
    `pipe_to_avatar` is async, so the callback builds a coroutine and drops it —
    it is never awaited and no frames are ever produced. Bug #1.

  * The spec fired every chunk's TTS concurrently and emitted on completion.
    TTS for "world." can finish before "Hello,", so audio plays out of order
    and speech is garbled. Bug #2.

The fix keeps full pipelining — chunk N+1's TTS runs while chunk N is being
rendered — but a single consumer AWAITS the TTS tasks in submission order, so
output is always ordered. A semaphore bounds in-flight TTS, giving backpressure
all the way up to LLM token consumption. One consumer means one writer to the
avatar/WebRTC tracks (no interleaved frames).

This module is transport- and model-agnostic (callables are injected) so it runs
and self-tests on a laptop with no GPU:  python3 pipeline.py
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Optional

# --------------------------------------------------------------------------- #
# Chunking buffer
# --------------------------------------------------------------------------- #
# A paralinguistic tag, e.g. [laugh], [sigh], [clear throat].
_TAG = r"\[[a-z][a-z ]*\]"
# Any candidate boundary: sentence-final punctuation, a complete tag, or a soft
# comma/semicolon/colon.
_BOUNDARY = re.compile(rf"[.!?]+|{_TAG}|[,;:]")


class Chunker:
    """Accumulates streamed tokens and emits speakable chunks at boundaries.

    Sentence ends (. ? !) and paralinguistic tags ALWAYS cut. A comma (or ; :)
    cuts only once the chunk already has >= min_words words — so we emit at
    commas for low latency on long clauses, but never produce tiny fragments.
    The first chunk of a turn uses a smaller minimum so the avatar starts
    speaking sooner (low time-to-first-frame).
    """

    def __init__(self, min_words: int = 4, first_min_words: int = 2):
        self.buf = ""
        self.first = True
        self.min_words = min_words
        self.first_min_words = first_min_words

    @staticmethod
    def _eat_spaces(s: str, end: int) -> int:
        while end < len(s) and s[end].isspace():
            end += 1
        return end

    def _cut(self) -> Optional[int]:
        min_w = self.first_min_words if self.first else self.min_words
        for m in _BOUNDARY.finditer(self.buf):
            end = m.end()
            is_soft = m.group()[0] in ",;:"
            if is_soft and len(self.buf[:end].split()) < min_w:
                continue  # comma too early — wait for more words or the next boundary
            return self._eat_spaces(self.buf, end)
        return None

    def feed(self, token: str) -> list[str]:
        """Add a token; return any chunks that are now complete."""
        self.buf += token
        out: list[str] = []
        while True:
            end = self._cut()
            if end is None:
                break
            chunk = self.buf[:end].strip()
            self.buf = self.buf[end:]
            if chunk:
                out.append(chunk)
                self.first = False
        return out

    def flush(self) -> Optional[str]:
        """Return whatever remains at end of stream (no trailing boundary)."""
        chunk = self.buf.strip()
        self.buf = ""
        self.first = False
        return chunk or None


# --------------------------------------------------------------------------- #
# Ordered streaming pipeline
# --------------------------------------------------------------------------- #
# Injected callables:
LLMStream = Callable[[str], AsyncIterator[str]]   # user_text -> async token stream
SynthFn = Callable[[str], Awaitable[object]]      # text chunk -> audio (bytes/obj)
OnAudio = Callable[[object, str], Awaitable[None]]  # ordered (audio, text) -> sink


@dataclass
class Pipeline:
    llm_stream: LLMStream
    synth: SynthFn
    on_audio: OnAudio
    max_inflight_tts: int = 3   # lookahead depth = backpressure bound

    async def run(self, user_text: str) -> None:
        sem = asyncio.Semaphore(self.max_inflight_tts)
        queue: asyncio.Queue = asyncio.Queue()

        async def submit(chunk: str) -> None:
            await sem.acquire()  # blocks (backpressure) when too many TTS in flight

            async def _synth() -> object:
                try:
                    return await self.synth(chunk)
                finally:
                    sem.release()

            await queue.put((chunk, asyncio.create_task(_synth())))

        async def produce() -> None:
            chunker = Chunker()
            try:
                async for token in self.llm_stream(user_text):
                    for chunk in chunker.feed(token):
                        await submit(chunk)
                tail = chunker.flush()
                if tail:
                    await submit(tail)
            finally:
                await queue.put(None)  # sentinel even if the LLM stream errors

        async def consume() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    break
                chunk, task = item
                audio = await task          # <-- ORDERED await => ordered playout
                await self.on_audio(audio, chunk)

        # Surface producer exceptions instead of silently hanging the consumer.
        await asyncio.gather(produce(), consume())


# --------------------------------------------------------------------------- #
# Self-test (no GPU): proves ordering holds even when TTS completes out of order
# --------------------------------------------------------------------------- #
async def _selftest() -> None:
    # 1) Chunker: streamed char-by-char should yield ordered, sensible chunks.
    text = "Hey there, welcome back. How are you today? [laugh] Glad to see you!"
    ch = Chunker()
    chunks: list[str] = []
    for c in text:
        chunks.extend(ch.feed(c))
    tail = ch.flush()
    if tail:
        chunks.append(tail)
    print("chunks:", chunks)
    assert "".join(chunks).replace(" ", "") == text.replace(" ", ""), \
        "chunker dropped/duplicated content"
    assert len(chunks) >= 4, "expected the turn to split into several chunks"

    # comma chunking: split at a comma once min_words is reached (not before)
    ch2 = Chunker()
    out2: list[str] = []
    for c in "Yes. I know a lot about movies, and comedies are my favorite.":
        out2.extend(ch2.feed(c))
    tail2 = ch2.flush()
    if tail2:
        out2.append(tail2)
    print("comma chunks:", out2)
    assert any(c.endswith("movies,") for c in out2), "should split at a comma once long enough"
    assert all(len(c.split()) >= 2 or c.endswith((".", "!", "?")) or c.startswith("[") for c in out2), \
        "no tiny comma fragments"

    # 2) Pipeline ordering: make EARLIER chunks take LONGER to synthesize, so they
    #    complete in reverse order. Correct output must still be submission order.
    order_in: list[str] = []
    order_out: list[str] = []

    async def fake_llm(_: str):
        for tok in ["One. ", "Two. ", "Three. ", "Four. ", "Five."]:
            await asyncio.sleep(0.005)
            yield tok

    n = {"i": 0}

    async def fake_synth(chunk: str):
        i = n["i"]
        n["i"] += 1
        order_in.append(chunk)
        # earlier chunk -> longer delay -> finishes LAST if unordered
        await asyncio.sleep(0.10 - i * 0.015)
        return f"audio({chunk})"

    async def fake_sink(audio, chunk):
        order_out.append(chunk)

    await Pipeline(fake_llm, fake_synth, fake_sink, max_inflight_tts=5).run("hi")
    print("submitted:", order_in)
    print("emitted:  ", order_out)
    assert order_out == order_in, f"OUT OF ORDER: {order_out} != {order_in}"
    print("\nOK — chunking correct and playout strictly ordered despite reversed TTS completion.")


if __name__ == "__main__":
    asyncio.run(_selftest())
