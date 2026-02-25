# Moonshine STT Server

Streaming speech-to-text server using [Moonshine](https://github.com/usefulsensors/moonshine) models, implementing the LocalGPT voice protocol over WebSocket.

## Architecture

```
  Browser / LocalGPT
       |
       | WebSocket (ws://0.0.0.0:8766/ws)
       | Binary PCM f32 LE, 16 kHz mono
       v
  +------------------+
  |   server.py      |
  |  FastAPI + WS    |
  |  Energy VAD      |
  +--------+---------+
           |
           v
  +------------------+
  | Moonshine        |
  | Transcriber      |
  | (global singleton) |
  +------------------+
           |
           v
  JSON events back to client:
    speech_start -> partial* -> final -> speech_end
```

## Setup

```bash
pip install -r requirements.txt
```

Moonshine models are downloaded automatically on first use by `moonshine-voice`. Ensure you have an internet connection for the initial run.

## Run

```bash
python server.py
```

The server starts on `ws://0.0.0.0:8766/ws` by default.

Open `http://localhost:8766` in a browser for the demo UI.

### Environment Variables

| Variable    | Default                     | Description          |
|-------------|-----------------------------|----------------------|
| `PORT`      | `8766`                      | Server port          |
| `STT_MODEL` | `moonshine-tiny-streaming`  | Moonshine model name |
| `LOG_LEVEL` | `INFO`                      | Logging level        |

## LocalGPT Configuration

Add to your `~/.localgpt/config.toml`:

```toml
[voice.stt]
provider = "ws"

[voice.stt.ws]
endpoint = "ws://127.0.0.1:8766/ws"
reconnect_interval_ms = 1000
max_reconnect_attempts = 10
```

## Protocol

The server follows the LocalGPT voice protocol:

- **Client -> Server**: Binary frames containing PCM float32 little-endian audio at 16 kHz mono
- **Server -> Client**: JSON text frames:

```jsonc
{"type": "speech_start", "timestamp_ms": 12340}
{"type": "partial",      "text": "hello"}
{"type": "final",        "text": "hello world", "language": "en", "confidence": 0.9, "duration_ms": 1200.0}
{"type": "speech_end",   "timestamp_ms": 13540, "duration_ms": 1200.0}
```

## Available Models

- `moonshine-tiny-streaming` (default, fastest)
- `moonshine-small-streaming` (more accurate)
