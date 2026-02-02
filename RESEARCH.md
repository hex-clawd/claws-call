# Telegram Real-Time Voice Call System — Research Document

*Date: 2026-02-02*

---

## 1. Telegram Bot Voice Call API — Can Bots Make Calls?

### The Hard Truth: Bots CANNOT make voice calls

**Telegram Bot API does NOT support voice/video calls.** Bots can send voice messages, audio files, and voice notes — but they cannot initiate, receive, or participate in voice calls or voice chats.

### The Solution: Userbot via MTProto

To do voice calls programmatically, you need a **userbot** (a regular Telegram account automated via MTProto), not a bot account.

### Key Library: **MarshalX/tgcalls**
- **Repo:** https://github.com/MarshalX/tgcalls
- **What it is:** Python bindings (via pybind11) to Telegram's official `tgcalls` C++ library (same WebRTC stack used in official clients)
- **Two parts:**
  - `tgcalls` — C++ Python extension
  - `pytgcalls` — High-level SDK using Pyrogram or Telethon as MTProto backend
- **Supports:**
  - ✅ Group voice chats (join, play, record)
  - ✅ Private incoming/outgoing 1-on-1 calls (in dev branch, "working but not in release")
  - ✅ Raw PCM audio I/O (16-bit, 48kHz)
  - ✅ arm64/Unix (Mac Mini M4 compatible)
  - ✅ File, Device, and Raw data transfer modes

### GroupCallRaw — The Key Class

```python
from pytgcalls.implementation import GroupCallRaw

# Callbacks give you raw PCM bytes
def on_recorded_data(group_call, data: bytes, length: int):
    # `data` = raw PCM 16-bit 48kHz from the call
    pass

def on_played_data(group_call, length: int) -> bytes:
    # Return `length` bytes of PCM audio to play into call
    return audio_bytes
```

This is the critical interface — you get raw audio bytes in and push raw audio bytes out.

### Private Calls Status
- Private 1-on-1 calls are listed as "TODO — already working but not in release"
- The `dev` branch has this functionality
- For a voice assistant, **group voice chat in a private group** is a viable workaround:
  1. Create a private group with just the userbot + the user
  2. Start a voice chat
  3. User joins the voice chat
  4. Full duplex audio streaming works

### What You Need
- A Telegram **user account** (not bot) — phone number required
- API credentials from https://my.telegram.org
- Pyrogram or Telethon as MTProto client
- `pip install pytgcalls[pyrogram]`

---

## 2. Streaming Audio — Getting Continuous Audio from a Telegram Call

### How it works with pytgcalls

**Three modes of audio I/O:**

| Mode | Class | Use Case |
|------|-------|----------|
| **Raw** | `GroupCallRaw` | Direct bytes in Python — **best for our use case** |
| **File** | `GroupCallFile` | Play/record to .raw files (including named pipes/FIFOs) |
| **Device** | `GroupCallDevice` | Virtual system audio devices |

### Raw Mode (recommended)
```python
group_call = GroupCallFactory(app).get_raw_group_call(
    on_played_data=my_playout_callback,
    on_recorded_data=my_record_callback
)
```

- **Format:** PCM 16-bit signed, 48000 Hz, mono
- **Callback frequency:** ~every 10-20ms (WebRTC frame size)
- Audio arrives in small chunks — perfect for streaming pipeline

### FIFO Pipe Alternative
If raw callbacks are tricky, you can use named pipes:
```bash
mkfifo /tmp/tg_input.raw
mkfifo /tmp/tg_output.raw
```
Then point `GroupCallFile` at these pipes. Other processes can read/write them.

---

## 3. Local STT — Real-Time Whisper Transcription

### whisper.cpp
- **Repo:** https://github.com/ggml-org/whisper.cpp
- Pure C/C++ port of OpenAI Whisper
- **Excellent Apple Silicon support** — uses Metal/ANE acceleration on M4
- Built-in `whisper-stream` example for real-time mic transcription

### whisper.cpp Stream Mode
```bash
# Build
git clone https://github.com/ggml-org/whisper.cpp
cd whisper.cpp && make

# Run real-time stream
./build/bin/whisper-stream -m models/ggml-base.en.bin -t 4 --step 500 --length 5000
```
- Samples audio every 500ms
- Runs inference on sliding 5-second window
- ~100ms inference time on Apple Silicon for base model

### ufal/whisper_streaming (Better for our use case)
- **Repo:** https://github.com/ufal/whisper_streaming
- Python library built on top of faster-whisper
- Designed specifically for **long-form real-time streaming**
- Uses local agreement algorithm to produce stable partial transcripts
- Supports multiple backends: faster-whisper, whisper.cpp, MLX whisper
- **Can accept audio from a pipe/stream**, not just microphone

```python
from whisper_online import FasterWhisperASR, OnlineASRProcessor

asr = FasterWhisperASR("en", "base")
online = OnlineASRProcessor(asr)

# Feed audio chunks as they arrive from Telegram
online.insert_audio_chunk(audio_np_array)
output = online.process_iter()  # Returns (start, end, text) or partial
```

### Performance on M4
- **whisper-tiny:** ~10x real-time (way faster than needed)
- **whisper-base:** ~5x real-time
- **whisper-small:** ~2-3x real-time
- **whisper-medium:** ~1x real-time (borderline)
- All with Metal acceleration. base or small is the sweet spot.

### Audio Format Bridge
- Telegram sends PCM 16-bit 48kHz
- Whisper expects 16kHz mono float32
- Simple conversion: downsample 48k→16k, convert int16→float32

---

## 4. Voice Activity Detection (VAD) / Turn Detection

### Silero VAD (Recommended)
- **Repo:** https://github.com/snakers4/silero-vad
- MIT license, no telemetry, no vendor lock
- PyTorch-based, works on CPU (fast on M4)
- **<1ms per audio chunk** on modern hardware
- Supports 16kHz and 8kHz
- Outputs speech probability per frame

```python
import torch
model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
(get_speech_timestamps, _, read_audio, *_) = utils

# For streaming, use per-chunk:
speech_prob = model(audio_chunk, 16000).item()
is_speech = speech_prob > 0.5
```

### Turn Detection Strategy
```
[silence] → [speech starts] → [speech continues] → [silence > 700ms] → TURN END
```

Parameters to tune:
- **Speech threshold:** 0.5 probability
- **End-of-turn silence:** 500-800ms (shorter = more responsive, more false triggers)
- **Min speech duration:** 300ms (ignore very short noise bursts)

### webrtcvad (Simpler alternative)
```python
import webrtcvad
vad = webrtcvad.Vad(2)  # aggressiveness 0-3
is_speech = vad.is_speech(frame_bytes, sample_rate=16000)
```
- Faster, simpler, less accurate than Silero
- No ML model needed

---

## 5. Interruption Handling

### The Problem
User starts speaking while TTS is still playing → need to:
1. Immediately stop sending TTS audio to the call
2. Discard remaining TTS buffer
3. Start capturing user's speech
4. (Optionally) cancel in-flight LLM generation

### Implementation
```python
class AudioPipeline:
    def __init__(self):
        self.tts_buffer = collections.deque()
        self.is_playing_tts = False
        self.interrupted = False
    
    def on_recorded_data(self, data):
        # VAD check on incoming audio
        if self.vad.is_speech(data) and self.is_playing_tts:
            # INTERRUPT!
            self.tts_buffer.clear()
            self.is_playing_tts = False
            self.interrupted = True
            # Cancel LLM generation if in progress
            self.cancel_llm()
        
        if not self.is_playing_tts:
            # Capture user speech for STT
            self.stt_buffer.append(data)
    
    def on_played_data(self, length):
        if self.tts_buffer:
            return self.tts_buffer.popleft()
        return b'\x00' * length  # silence
```

### Key insight
Since `GroupCallRaw` gives you both input AND output callbacks, you have full control. The VAD runs on the input, and you control what goes to the output buffer. Interruption is just "clear the output buffer."

---

## 6. TTS Streaming

### Edge TTS (Free, no API key)
- Microsoft Edge's free TTS service
- High-quality neural voices
- No API key required
- Python package: edge-tts
- Latency: ~300-500ms

```python
import edge_tts

async def generate_tts(text: str, output_path: str):
    communicate = edge_tts.Communicate(text, voice='en-GB-RyanNeural')
    await communicate.save(output_path)
```

### Local TTS Alternatives
- **Piper TTS** (https://github.com/rhasspy/piper) — Fast, local, decent quality. C++ with Python bindings. ~100ms latency.
- **Coqui/XTTS** — Higher quality, slower. GPU helps.
- **MLX-based TTS** — Emerging options for Apple Silicon.
- **macOS `say` command** — Terrible quality but zero setup.

### Audio Format Bridge (TTS → Telegram)
- Most TTS outputs 16kHz or 22kHz
- Telegram needs 48kHz PCM 16-bit
- Upsample with scipy/librosa or ffmpeg

---

## 7. Full Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Mac Mini M4                       │
│                                                      │
│  ┌──────────┐    MTProto     ┌───────────────────┐  │
│  │ Telegram  │◄─────────────►│ Pyrogram/Telethon │  │
│  │  Servers  │   (userbot)   │   + pytgcalls      │  │
│  └──────────┘                └─────┬───────┬─────┘  │
│                                    │       │         │
│                              record│  play │         │
│                           PCM 48k  │  PCM 48k       │
│                                    ▼       ▲         │
│                              ┌─────────────────┐    │
│                              │  Audio Pipeline  │    │
│                              │  (Python async)  │    │
│                              └──┬──────────┬───┘    │
│                                 │          │         │
│                          ┌──────▼──┐  ┌────┴─────┐  │
│                          │ Silero  │  │ TTS      │  │
│                          │ VAD     │  │ Buffer   │  │
│                          └──┬──────┘  └────▲─────┘  │
│                             │              │         │
│                      ┌──────▼──────┐  ┌────┴─────┐  │
│                      │ Resample    │  │ Resample │  │
│                      │ 48k → 16k  │  │ 16k→48k  │  │
│                      └──────┬──────┘  └────▲─────┘  │
│                             │              │         │
│                      ┌──────▼──────┐  ┌────┴─────┐  │
│                      │ Whisper STT │  │ Edge TTS │  │
│                      │(faster-wh.) │  │ or Piper │  │
│                      └──────┬──────┘  └────▲─────┘  │
│                             │              │         │
│                      ┌──────▼──────────────┴─────┐  │
│                      │        LLM (Claude)       │  │
│                      │     via API / streaming    │  │
│                      └───────────────────────────┘  │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Flow:
1. **User speaks** → Telegram WebRTC → pytgcalls `on_recorded_data` callback → PCM 48kHz bytes
2. **VAD** → Silero checks each chunk for speech activity
3. **When speech ends** (silence > 700ms) → accumulated audio sent to STT
4. **STT** → faster-whisper or whisper.cpp transcribes → text
5. **LLM** → Claude API with streaming response
6. **TTS** → Edge TTS (stream text as LLM generates) → audio chunks
7. **Play back** → Audio chunks pushed into `on_played_data` callback → Telegram WebRTC → user hears response

### Latency Budget (estimated)
| Step | Time |
|------|------|
| End-of-turn detection | ~700ms |
| STT (whisper base) | ~200-500ms |
| LLM first token | ~300-800ms |
| TTS first audio | ~300-500ms |
| **Total to first audio** | **~1.5-2.5s** |

This is acceptable for conversational AI. Comparable to phone-based assistants.

---

## 8. Existing Projects

### Direct matches (Telegram voice assistant)
- **None found** that do real-time voice call + STT + LLM + TTS over Telegram calls.
- Most "Telegram voice bots" just transcribe voice messages (batch, not real-time).

### Related/Useful
| Project | What it does | Link |
|---------|-------------|------|
| **MarshalX/tgcalls** | Core library for Telegram calls | https://github.com/MarshalX/tgcalls |
| **pytgcalls** | High-level Python SDK | https://pypi.org/project/pytgcalls/ |
| **Pipecat** (Daily.co) | Open-source voice agent framework | https://github.com/pipecat-ai/pipecat |
| **ufal/whisper_streaming** | Real-time streaming STT | https://github.com/ufal/whisper_streaming |
| **Silero VAD** | Voice activity detection | https://github.com/snakers4/silero-vad |
| **Piper TTS** | Fast local TTS | https://github.com/rhasspy/piper |
| **Edge TTS** | Free Microsoft TTS | https://github.com/rany2/edge-tts |
| **TheHamkerCat/Telegram_VC_Bot** | Music bot for voice chats | https://github.com/TheHamkerCat/Telegram_VC_Bot |

### Pipecat — Worth Studying
Pipecat by Daily.co is the closest architectural reference. It's a full voice agent framework with:
- STT (Whisper, Deepgram, etc.)
- LLM (OpenAI, Anthropic, etc.)
- TTS (Edge TTS, etc.)
- VAD (Silero built-in)
- Interruption handling
- Transport layer (Daily rooms, WebRTC)

It doesn't have a Telegram transport, but the pipeline architecture is exactly what we'd build. We could potentially write a Telegram transport adapter for Pipecat, or just steal their architecture patterns.

---

## 9. Platform Constraints

### Userbot vs Bot — MUST use Userbot
| Feature | Bot API | Userbot (MTProto) |
|---------|---------|-------------------|
| Voice calls | ❌ No | ✅ Yes |
| Voice chats | ❌ No | ✅ Yes |
| Raw audio stream | ❌ No | ✅ Yes |
| Voice messages | ✅ Yes | ✅ Yes |
| Phone number needed | No | **Yes** |
| Rate limits | Bot limits | User limits |
| Risk of ban | None | **Possible** (automated userbot) |

### Ban Risk Mitigation
- Use a dedicated phone number/SIM for the userbot
- Don't spam — voice calls to a known user are fine
- Telegram doesn't actively ban voice-chat userbots (many music bots exist)
- Keep behavior human-like (don't join/leave hundreds of chats)

### Mac Mini M4 Compatibility
- ✅ whisper.cpp with Metal acceleration — excellent
- ✅ faster-whisper via CTranslate2 — works on ARM64
- ✅ Silero VAD — PyTorch on Apple Silicon, fast
- ✅ pytgcalls — supports arm64 Unix
- ✅ Piper TTS — ARM64 builds available
- ✅ Python 3.10+ ecosystem — all good

### Alternative Approach: Voice Messages (Simpler)
If real-time calls prove too complex, a simpler MVP:
1. User sends voice message → bot receives audio file
2. Transcribe with whisper → LLM → TTS
3. Send back as voice message
4. Latency: 3-8 seconds, but much simpler

---

## 10. Feasibility Assessment

### Verdict: **FEASIBLE but requires effort**

| Aspect | Difficulty | Notes |
|--------|-----------|-------|
| Telegram call integration | 🟡 Medium | pytgcalls is mature for voice chats; private calls are dev-only |
| Raw audio streaming | 🟢 Easy | GroupCallRaw provides clean PCM I/O |
| Real-time STT | 🟢 Easy | whisper.cpp/faster-whisper excellent on M4 |
| VAD / Turn detection | 🟢 Easy | Silero VAD is plug-and-play |
| Interruption handling | 🟡 Medium | Needs careful async design |
| TTS streaming | 🟢 Easy | Edge TTS or Piper |
| Full pipeline integration | 🟡 Medium | Async plumbing, format conversion, error handling |
| Private 1-on-1 calls | 🔴 Hard | Not in stable release; use voice chat workaround |

### Recommended Approach
1. **Phase 1 (MVP):** Voice messages — user sends voice msg → STT → LLM → TTS → voice msg reply. Build the STT/LLM/TTS pipeline first.
2. **Phase 2:** Group voice chat mode — create a private group, use pytgcalls with GroupCallRaw for real-time streaming.
3. **Phase 3:** Private calls — if/when MarshalX releases stable private call support, migrate.

### Estimated Development Time
- Phase 1: 1-2 days
- Phase 2: 3-5 days
- Phase 3: Depends on upstream library

### Key Dependencies to Install
```bash
pip install pyrogram tgcalls pytgcalls[pyrogram]
pip install faster-whisper  # or build whisper.cpp
pip install torch  # for Silero VAD
pip install edge-tts  # or piper-tts for local
pip install numpy scipy  # audio processing
```
