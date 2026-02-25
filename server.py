"""Moonshine STT WebSocket server implementing LocalGPT voice protocol."""

import asyncio
import json
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

PORT = int(os.environ.get("PORT", "8799"))
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

    # Client audio format (may be updated by config message)
    client_encoding = "pcm_f32le"  # default: raw f32
    client_sample_rate = SAMPLE_RATE

    try:
        while True:
            data = await ws.receive()

            # Handle text (JSON) messages — config or end_of_stream
            if "text" in data:
                try:
                    msg = json.loads(data["text"])
                except Exception:
                    logger.warning("non-JSON text message ignored: %s", data["text"][:100])
                    continue

                msg_type = msg.get("type", "")
                if msg_type == "config":
                    client_encoding = msg.get("encoding", "pcm_f32le")
                    client_sample_rate = int(msg.get("sample_rate", SAMPLE_RATE))
                    logger.info(
                        "Client config: encoding=%s sample_rate=%d",
                        client_encoding,
                        client_sample_rate,
                    )
                    await ws.send_json({"type": "config_ack"})
                    continue
                elif msg_type == "end_of_stream":
                    logger.info("Client sent end_of_stream")
                    # Flush remaining audio if in speech
                    if in_speech and len(audio_buffer) > 0:
                        now_ms = time.time() * 1000
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
                        in_speech = False
                        audio_buffer = np.empty(0, dtype=np.float32)
                    continue
                else:
                    logger.debug("ignoring text message type: %s", msg_type)
                    continue

            # Binary (audio) data
            if "bytes" not in data:
                continue
            raw = data["bytes"]

            # Decode PCM based on client encoding
            if client_encoding == "pcm_s16le":
                samples_i16 = np.frombuffer(raw, dtype=np.int16)
                samples = samples_i16.astype(np.float32) / 32768.0
            else:
                samples = np.frombuffer(raw, dtype=np.float32)

            # Resample if client sample rate differs from model sample rate
            if client_sample_rate != SAMPLE_RATE and len(samples) > 0:
                ratio = SAMPLE_RATE / client_sample_rate
                num_out = int(len(samples) * ratio)
                if num_out > 0:
                    indices = np.arange(num_out) / ratio
                    indices = np.clip(indices.astype(int), 0, len(samples) - 1)
                    samples = samples[indices]
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
