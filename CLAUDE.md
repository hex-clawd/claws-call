# Telegram Voice Call AI Assistant

Real-time voice conversation system over Telegram using local STT (whisper.cpp), Claude API, and ElevenLabs TTS.

## Architecture
Telegram voice chat (userbot/MTProto) → pytgcalls (raw PCM) → Silero VAD → whisper STT → Claude → ElevenLabs TTS → back to call

## Plan Mode
- Make the plan extremely concise. Sacrifice grammar for the sake of concision.
- At the end of each plan, give me a list of unresolved questions to answer, if any.

## Key Constraints
- Runs on Mac Mini M4 (Apple Silicon)
- Must use userbot (MTProto) — Telegram bots can't do voice calls
- Local whisper.cpp for STT (model at ~/clawd/models/ggml-small.en.bin)
- Target latency: <2.5s end-to-end
- Workaround for private calls: 2-person private group → voice chat

## Telegram Voice Chat Timeout Behavior

**Problem:** Telegram voice chats disconnect if they don't receive audio frames for too long (~few hundred ms). During LLM API calls (which can take several seconds when overloaded), no audio is being generated, causing disconnects.

**Solution:** Two-layer defense in `VoiceChat`:

1. **`_playback_loop`** (PRIMARY) - Sends real audio when buffer has data, OR sends silence every 30ms when buffer is empty. This is the main frame sender.
2. **`_keepalive_loop`** (BACKUP) - Independent task that sends silence every 40ms if `_last_frame_time` is stale. Catches any gaps the playback loop misses.

**Key insight:** During LLM streaming, `is_speaking=True` but `output_buffer` is empty — the bot is "thinking" but not generating audio yet. The playback loop's silence-sending handles this state (sends silence every 30ms), with keepalive as backup.

**Why two layers?** If the playback loop gets blocked by heavy async work (LLM streaming, TTS processing), the keepalive loop provides a safety net. Both tasks update `_last_frame_time` so they don't duplicate frames.

## Current Status (2026-02-03)

### The Triangle of Problems
We've been cycling through three interconnected issues:

1. **TIMEOUT/DISCONNECT** - Bot stops sending frames (even silence), Telegram times out and disconnects
   - Happens when: Waiting for LLM response, heavy async processing
   - Root cause: Playback loop only sends silence when `output_buffer` is empty, but during LLM wait the state is ambiguous
   - **FIXED (2026-02-03):** Playback loop now sends silence every 30ms when buffer empty, plus keepalive backup at 40ms

2. **WON'T INTERRUPT** - Bot doesn't stop talking when user starts speaking
   - Happens when: User interrupts mid-TTS playback
   - Root cause: VAD check was removed because it blocked event loop (problem #3)
   - Status: Sampled VAD check every 10 frames (~100ms) implemented in frame handler

3. **OVERLOADS EVENT LOOP** - Too much processing blocks Telegram signaling, user can't even connect mic
   - Happens when: VAD runs on every frame (100x/sec), or too many silence frames sent
   - Root cause: Silero VAD inference is synchronous, takes 5-20ms per call
   - Status: Using sampled VAD (every 10 frames) to balance responsiveness vs CPU

### The Balance We Need
- Send silence often enough to prevent timeout (~every 30ms) ✅
- Check VAD often enough for responsive interrupts (~every 100ms) ✅
- But NOT so often that it blocks the event loop ✅

### What's Implemented
- `voice_chat.py`: Playback loop sends silence every 30ms when buffer empty (PRIMARY)
- `voice_chat.py`: Keepalive loop as backup safety net every 40ms (BACKUP)
- `voice_chat.py`: Sampled interrupt check every 10 frames with dedicated VAD instance
- `voice_pipeline.py`: Async interrupt detection in processing loop

### Next Steps
1. ~~Ensure silence keepalive runs INDEPENDENTLY of playback state~~ ✅ Done
2. Test sampled interrupt (currently every 10 frames = ~100ms)
3. Consider `run_in_executor()` for VAD to make it truly non-blocking

## See RESEARCH.md for full technical research
