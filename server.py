"""Moonshine STT WebSocket server implementing LocalGPT voice protocol."""

import asyncio
import logging
import os
import time

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PORT = int(os.environ.get("PORT", "8766"))
STT_MODEL = os.environ.get("STT_MODEL", "moonshine-tiny-streaming")
MODEL_ARCH = os.environ.get("MODEL_ARCH", "TINY").upper()
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
VERSION = "0.1.0"

RMS_THRESHOLD = 0.02
SILENCE_FRAMES_LIMIT = 30  # ~30 frames of silence before speech_end
SAMPLE_RATE = 16000

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load model once at startup
# ---------------------------------------------------------------------------

try:
    from moonshine_voice import Transcriber, ModelArch

    _model_arch = ModelArch.BASE if MODEL_ARCH == "BASE" else ModelArch.TINY
    _transcriber = Transcriber(STT_MODEL, model_arch=_model_arch)
    logger.info("Using model arch: %s", MODEL_ARCH)
    logger.info("Loaded moonshine model: %s", STT_MODEL)
except ImportError:
    _transcriber = None
    logger.error("moonshine_voice not installed – transcription unavailable")

_transcriber_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Moonshine STT Server", version=VERSION)


@app.get("/health")
async def health():
    return JSONResponse({"status": "ok", "model": STT_MODEL, "version": VERSION})


@app.get("/models")
async def models():
    return JSONResponse(
        {"models": ["moonshine-tiny-streaming", "moonshine-small-streaming"]}
    )


@app.get("/")
async def index():
    return FileResponse("index.html", media_type="text/html")


# ---------------------------------------------------------------------------
# VAD helpers
# ---------------------------------------------------------------------------


def rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(samples**2)))


# ---------------------------------------------------------------------------
# Transcription helper
# ---------------------------------------------------------------------------


def _do_transcribe(audio: np.ndarray) -> str:
    """Run transcription using the global model (called in executor)."""
    try:
        result = _transcriber.transcribe_without_streaming(audio, SAMPLE_RATE)
        return " ".join(l.text for l in result.lines)
    except AttributeError:
        return _transcriber.transcribe(audio)


# ---------------------------------------------------------------------------
# WebSocket endpoint
# ---------------------------------------------------------------------------


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("Client connected")

    if _transcriber is None:
        await ws.close(code=1011, reason="moonshine_voice not available")
        return

    loop = asyncio.get_running_loop()

    in_speech = False
    silence_frames = 0
    speech_start_ms: float = 0.0
    audio_buffer = np.empty(0, dtype=np.float32)
    last_partial_len = 0

    try:
        while True:
            data = await ws.receive_bytes()

            # Decode PCM f32 LE
            samples = np.frombuffer(data, dtype=np.float32)
            level = rms(samples)
            now_ms = time.time() * 1000

            if not in_speech:
                if level >= RMS_THRESHOLD:
                    in_speech = True
                    silence_frames = 0
                    speech_start_ms = now_ms
                    audio_buffer = samples.copy()
                    last_partial_len = 0

                    await ws.send_json(
                        {"type": "speech_start", "timestamp_ms": speech_start_ms}
                    )
                    logger.debug("speech_start at %.0f", speech_start_ms)
            else:
                # Accumulate audio
                audio_buffer = np.concatenate([audio_buffer, samples])

                if level < RMS_THRESHOLD:
                    silence_frames += 1
                else:
                    silence_frames = 0

                # Emit partial transcription periodically
                if len(audio_buffer) - last_partial_len >= SAMPLE_RATE // 2:
                    async with _transcriber_lock:
                        partial_text = await loop.run_in_executor(
                            None, _do_transcribe, audio_buffer
                        )
                    last_partial_len = len(audio_buffer)
                    if partial_text:
                        await ws.send_json(
                            {"type": "partial", "text": partial_text}
                        )

                # End of speech
                if silence_frames >= SILENCE_FRAMES_LIMIT:
                    duration_ms = now_ms - speech_start_ms

                    async with _transcriber_lock:
                        final_text = await loop.run_in_executor(
                            None, _do_transcribe, audio_buffer
                        )

                    language = getattr(_transcriber, "language", "en")

                    await ws.send_json(
                        {
                            "type": "final",
                            "text": final_text or "",
                            "language": language,
                            "confidence": 0.9,
                            "duration_ms": duration_ms,
                        }
                    )
                    await ws.send_json(
                        {
                            "type": "speech_end",
                            "timestamp_ms": now_ms,
                            "duration_ms": duration_ms,
                        }
                    )
                    logger.info(
                        "final: %r (%.0fms)", final_text, duration_ms
                    )

                    # Reset state
                    in_speech = False
                    silence_frames = 0
                    audio_buffer = np.empty(0, dtype=np.float32)
                    last_partial_len = 0

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception:
        logger.exception("WebSocket error")
    finally:
        logger.info("Connection closed")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL.lower())
