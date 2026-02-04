# Telegram Voice Call AI Assistant

Real-time voice conversation system over Telegram with two operational modes:
- **Phase 1**: Voice message-based interaction (stable)
- **Phase 2**: Real-time voice chat with live conversation (NEW!)

## Features

### Phase 1: Voice Messages
- Receives voice messages from authorized Telegram user
- Transcribes audio using faster-whisper (local STT)
- Generates responses using Claude API
- Converts responses to speech using Edge TTS
- Replies with voice messages

### Phase 2: Real-Time Voice Chat (NEW!)
- Live bidirectional voice conversation in Telegram group voice chats
- Voice Activity Detection (Silero VAD) for turn detection
- Interruption handling: cancel AI response when user speaks
- Streaming TTS for lower latency
- Auto-rejoin on disconnect with exponential backoff
- Configurable VAD thresholds for tuning responsiveness

## Architecture

### Phase 1: Voice Messages
```
Voice Message → STT (faster-whisper) → Claude API → TTS (Edge TTS) → Voice Reply
```

### Phase 2: Real-Time Voice Chat
```
Telegram Voice Chat (pytgcalls GroupCallRaw)
    ↓ PCM 48kHz
Audio Pipeline
    ↓ Resample to 16kHz
Silero VAD (turn detection)
    ↓ On turn end
faster-whisper STT
    ↓ Transcription
Claude API (streaming capable)
    ↓ Response text
Edge TTS (streaming)
    ↓ Resample to 48kHz
Back to Voice Chat
```

## Requirements

- Python 3.11+
- Mac Mini M4 (Apple Silicon) or compatible system
- Telegram account (phone number required)
- API keys: Telegram, Anthropic (Claude)
- For Phase 2: A private Telegram group (for voice chat)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Telegram API Credentials

1. Go to https://my.telegram.org
2. Log in with your phone number
3. Go to "API development tools"
4. Create a new application
5. Copy your `API_ID` and `API_HASH`

### 3. Get Your Telegram User ID

1. Message [@userinfobot](https://t.me/userinfobot) on Telegram
2. Copy your user ID (numeric)

### 4. Get Claude API Key

1. Go to https://console.anthropic.com/
2. Create an API key
3. Copy the key (starts with `sk-ant-`)

### 5. Configure Environment

Create a `.env` file from the template:

```bash
cp .env.example .env
```

Edit `.env` and fill in all the values:

```bash
# Telegram Userbot
API_ID=12345678                              # From my.telegram.org
API_HASH=abcdef1234567890abcdef1234567890   # From my.telegram.org
PHONE_NUMBER=+1234567890                     # Your phone number with country code

# Security: Only this user can interact with the bot
AUTHORIZED_USER_ID=123456789                 # From @userinfobot

# Claude
ANTHROPIC_API_KEY=sk-ant-...                 # From console.anthropic.com

# Whisper
WHISPER_MODEL=small.en                       # Model size: tiny.en, base.en, small.en, medium.en

# Bot Mode
BOT_MODE=voice_message                       # "voice_message" (Phase 1) or "voice_chat" (Phase 2)

# Voice Chat Settings (required for voice_chat mode)
VOICE_CHAT_GROUP_ID=0                        # Private group chat ID (see setup below)

# VAD Settings (for voice_chat mode)
VAD_SILENCE_THRESHOLD_MS=700                 # Silence duration to detect end of turn
VAD_SPEECH_THRESHOLD=0.5                     # Speech detection sensitivity
VAD_MIN_SPEECH_DURATION_MS=300               # Minimum speech duration

# Logging
LOG_LEVEL=INFO
```

### 6. Clawdbot Gateway Setup (Optional)

If you're running [Clawdbot](https://github.com/clawdbot/clawdbot), you can route voice conversations through your Clawdbot agent instead of calling Claude API directly. This gives you:

- Conversations go through your agent with full context
- Your agent's personality and memory
- Integration with other Clawdbot features

**Setup:**

1. **Get your Gateway Token** from Clawdbot's config:
   ```bash
   cat ~/.clawdbot/clawdbot.json | grep -A2 '"auth"' | grep token
   ```
   Or check your Clawdbot config file at `~/.clawdbot/clawdbot.json` under `gateway.auth.token`.

2. **Add to your `.env`:**
   ```bash
   CLAWDBOT_GATEWAY_URL=ws://127.0.0.1:18789
   CLAWDBOT_GATEWAY_TOKEN=your_token_here
   ```

3. **Ensure Clawdbot is running** — the gateway must be active for voice chat to work.

**Note:** The gateway runs on localhost only (`127.0.0.1`), so Clawdbot must be on the same machine as this voice bot.

**Without Clawdbot:** If you don't set the gateway token, the bot will fall back to direct Claude API calls using `ANTHROPIC_API_KEY`.

## Usage

### First Run

On first run, you'll need to authenticate with Telegram:

```bash
python main.py
```

You'll receive a code via Telegram. Enter it when prompted. A session file will be created for future runs.

### Phase 1: Voice Message Mode

Set `BOT_MODE=voice_message` in `.env`, then:

```bash
python main.py
```

The bot will start and listen for voice messages.

**Interacting:**
1. Open Telegram and find your own chat (Saved Messages) or any private chat
2. Send a voice message
3. The bot will:
   - Transcribe your message
   - Generate a response using Claude
   - Convert it to speech
   - Reply with a voice message

### Phase 2: Voice Chat Mode (Real-Time)

**Setup:**

1. Create a private group in Telegram:
   - Create a new group
   - Add only yourself and the bot account
   - Start a voice chat in the group

2. Get the group chat ID:
   ```python
   # Run this script or use @RawDataBot
   from pyrogram import Client
   app = Client("userbot", api_id=API_ID, api_hash=API_HASH)
   app.start()

   # List all dialogs to find your group
   for dialog in app.get_dialogs():
       print(f"{dialog.chat.title}: {dialog.chat.id}")
   ```

3. Set configuration in `.env`:
   ```bash
   BOT_MODE=voice_chat
   VOICE_CHAT_GROUP_ID=-1001234567890  # Your group chat ID (negative number)
   ```

4. Run the bot:
   ```bash
   python main.py
   ```

5. Join the voice chat in Telegram (the bot will already be there)

**Interacting:**
1. Speak naturally into the voice chat
2. The bot detects when you finish speaking (700ms silence by default)
3. Your speech is transcribed, sent to Claude, and the response is spoken back
4. You can interrupt the bot at any time by speaking

**Tuning VAD:**
- Adjust `VAD_SILENCE_THRESHOLD_MS` for faster/slower turn detection
- Lower = more responsive but may cut off mid-sentence
- Higher = more patient but slower response
- Default 700ms works well for most cases

## Security

- **Hardcoded authorization**: Only the user ID specified in `AUTHORIZED_USER_ID` can interact with the bot
- All other users are silently ignored
- No authentication bypass possible

## Project Structure

```
telegram-voice-call/
├── main.py                 # Entry point (supports both modes)
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (gitignored)
├── .env.example            # Environment template
├── PLAN.md                 # Implementation plan
├── RESEARCH.md             # Technical research
├── bot/
│   ├── __init__.py
│   ├── userbot.py          # Pyrogram userbot with voice message handler
│   └── voice_chat.py       # pytgcalls integration for real-time voice chat
├── audio/
│   ├── __init__.py
│   ├── stt.py              # Speech-to-text (faster-whisper)
│   ├── tts.py              # Text-to-speech (Edge TTS with streaming)
│   ├── vad.py              # Voice Activity Detection (Silero VAD)
│   └── utils.py            # Audio resampling and format conversion
├── llm/
│   ├── __init__.py
│   └── claude.py           # Claude API integration
└── pipeline/
    ├── __init__.py
    └── voice_pipeline.py   # Main async pipeline for real-time processing
```

## Troubleshooting

### Authentication Issues

If you get errors about API_ID or API_HASH:
- Verify they are correct in `.env`
- Make sure there are no extra spaces or quotes

### Whisper Model Issues

If faster-whisper fails to load:
- Try a smaller model: `WHISPER_MODEL=base.en` or `tiny.en`
- Ensure you have enough disk space for model download

### Voice Message Not Processing

- Check that you're messaging from the authorized user account
- Check logs in `bot.log` for errors
- Verify all API keys are valid

### TTS/STT Errors

- Verify API keys are correct
- Check internet connection
- Check API quotas/limits

## Implementation Status

- ✅ Phase 1: Voice message MVP (complete)
- ✅ Phase 2: Real-time voice chat with VAD and interruption handling (complete)
- ⏳ Phase 3: Latency optimization and polish (planned)

See `PLAN.md` for full roadmap.

## License

MIT

## Support

For issues or questions, check the logs in `bot.log` first.
