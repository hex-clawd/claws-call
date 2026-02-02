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

## See RESEARCH.md for full technical research
