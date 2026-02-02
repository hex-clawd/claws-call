# Implementation Plan — Telegram Voice Call AI Assistant

## Phases

### Phase 1: Voice Message MVP (2-3 days)
- User sends voice msg → bot transcribes → Claude → TTS → voice msg reply
- Validate STT/LLM/TTS pipeline, format conversions, API integrations
- No real-time complexity yet
- Use Bot API (simple)

### Phase 2: Group Voice Chat Real-Time (3-5 days)
- Userbot joins private group voice chat
- `GroupCallRaw` callbacks for PCM I/O
- Silero VAD → turn detection (700ms silence threshold)
- Streaming STT (whisper.cpp or faster-whisper)
- Claude API streaming
- ElevenLabs WebSocket TTS streaming
- Async pipeline with interruption handling

### Phase 3: Polish (1-2 days)
- Latency optimization (<2.5s target)
- Error recovery, reconnection logic
- Logging, monitoring
- Private 1-on-1 calls if `tgcalls` releases stable support

---

## Tech Stack

### Telegram
- **Userbot:** Pyrogram (cleaner API than Telethon)
- **Voice calls:** `pytgcalls[pyrogram]` v3.x with `GroupCallRaw`
- Session file for auth persistence

### STT
- **Option A (preferred):** `faster-whisper` + `whisper_streaming` (ufal)
  - Python-native, streaming-ready
  - Model: `base.en` or `small.en` (already have ggml at `~/clawd/models/ggml-small.en.bin`)
- **Option B:** whisper.cpp via subprocess
  - Lower overhead, but harder to integrate streaming

### VAD
- **Silero VAD** via `torch.hub.load('snakers4/silero-vad', 'silero_vad')`
- Threshold: 0.5 speech prob
- End-of-turn: 700ms silence

### LLM
- **Claude API** (Anthropic SDK)
- Streaming messages

### TTS
- **ElevenLabs** WebSocket streaming (primary)
- **Fallback:** Piper TTS (local, fast, free)
- Output: 16kHz PCM → resample to 48kHz for Telegram

### Audio Processing
- `numpy`, `scipy` for resampling
- `pydub` or raw `struct` for format conversions
- `asyncio` for concurrent I/O

### Environment
- Python 3.11+ (M4-optimized)
- PyTorch with MPS backend for Silero VAD

---

## File Structure

```
telegram-voice-call/
├── .env.example           # API keys template
├── .env                   # API keys (gitignored)
├── CLAUDE.md              # Project instructions
├── RESEARCH.md            # Technical research
├── PLAN.md                # This file
├── requirements.txt       # Python deps
├── main.py                # Entry point
├── config.py              # Load env vars
├── bot/
│   ├── __init__.py
│   ├── userbot.py         # Pyrogram client setup
│   └── voice_chat.py      # pytgcalls GroupCallRaw integration
├── audio/
│   ├── __init__.py
│   ├── vad.py             # Silero VAD turn detection
│   ├── stt.py             # Whisper STT (faster-whisper or whisper_streaming)
│   ├── tts.py             # ElevenLabs + Piper TTS
│   └── utils.py           # Resample, format conversion
├── llm/
│   ├── __init__.py
│   └── claude.py          # Claude API streaming
└── pipeline/
    ├── __init__.py
    └── voice_pipeline.py  # Main async pipeline, interruption handling
```

---

## API Keys / Accounts Needed

### 1. Telegram Userbot
- **Phone number** (dedicated SIM recommended)
- **API credentials:** Get from https://my.telegram.org
  - `API_ID` (int)
  - `API_HASH` (str)
- **Session file:** Generated on first login via Pyrogram
  - Stored as `userbot.session` (gitignored)

### 2. Claude API
- **API key:** https://console.anthropic.com/
  - `ANTHROPIC_API_KEY`
- Model: `claude-3-5-sonnet-20241022` or latest

### 3. ElevenLabs
- **API key:** https://elevenlabs.io/app/settings/api-keys
  - `ELEVENLABS_API_KEY`
- **Voice ID:** Choose from voice library
  - `ELEVENLABS_VOICE_ID`

### 4. (Optional) Piper TTS
- No API key needed (local)
- Download model from https://github.com/rhasspy/piper/releases

---

## `.env.example`
```bash
# Telegram Userbot
API_ID=12345678
API_HASH=abcdef1234567890abcdef1234567890
PHONE_NUMBER=+1234567890

# Claude
ANTHROPIC_API_KEY=sk-ant-...

# ElevenLabs
ELEVENLABS_API_KEY=...
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM  # Rachel

# Whisper
WHISPER_MODEL_PATH=~/clawd/models/ggml-small.en.bin

# Optional: Piper TTS fallback
PIPER_MODEL_PATH=~/models/piper/en_US-lessac-medium.onnx
```

---

## Implementation Details

### Audio Flow (Phase 2)
```python
# pseudocode
async def on_recorded_data(call, pcm_48k: bytes):
    # 1. Resample 48k → 16k
    pcm_16k = resample(pcm_48k, 48000, 16000)

    # 2. VAD check
    is_speech = vad.check(pcm_16k)

    if is_speech and ai_is_speaking:
        # INTERRUPT: clear TTS buffer, stop playback
        interrupt()

    if is_speech:
        stt_buffer.append(pcm_16k)
    elif stt_buffer and silence_duration > 0.7:
        # Turn complete
        text = await stt.transcribe(stt_buffer)
        await handle_user_turn(text)

async def on_played_data(call, length: int) -> bytes:
    if tts_buffer:
        chunk = tts_buffer.popleft()
        # Resample 16k → 48k if needed
        return resample(chunk, 16000, 48000)
    return b'\x00' * length  # silence

async def handle_user_turn(text: str):
    async for llm_chunk in claude.stream(text):
        async for audio_chunk in elevenlabs.stream(llm_chunk):
            tts_buffer.append(audio_chunk)
```

### Key Async Patterns
- `asyncio.Queue` for audio buffers
- `asyncio.Event` for interruption signaling
- `asyncio.create_task` for concurrent TTS generation + playback

### Latency Optimizations
- Start TTS as soon as LLM produces first sentence
- Pre-buffer first 500ms of TTS before playing
- Use `faster-whisper` with `beam_size=1` (greedy decoding, faster)
- Use `ufal/whisper_streaming` for incremental transcription

---

## Unresolved Questions

1. **Whisper backend:** Use `faster-whisper` (Python) or whisper.cpp (subprocess)? Leaning toward `faster-whisper` for streaming integration.

2. **Private group setup:** Auto-create group on first run, or manual setup? Auto-create risks Telegram spam detection. Leaning toward manual with instructions.

3. **VAD parameters:** 700ms silence threshold OK, or need adaptive (longer silence for long answers)? Start with 700ms, tune later.

4. **TTS fallback:** Always use ElevenLabs, or detect API failure → Piper fallback? Add fallback logic in Phase 3.

5. **Interruption UX:** Just stop TTS, or also cancel LLM generation? Cancel LLM to save costs + latency.

6. **Session persistence:** How to handle Pyrogram session file? Store as `userbot.session`, gitignore, document in README.

7. **Error recovery:** If voice chat disconnects, auto-rejoin? Yes, with exponential backoff.

8. **Multi-user:** Support multiple users in same group voice chat, or 1-on-1 only? Start with 1-on-1 (private group with 2 users).
