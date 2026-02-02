# Telegram Voice Call AI Assistant

Real-time voice conversation system over Telegram. Phase 1 MVP: Voice message-based interaction.

## Features (Phase 1)

- Receives voice messages from authorized Telegram user
- Transcribes audio using faster-whisper (local STT)
- Generates responses using Claude API
- Converts responses to speech using Edge TTS
- Replies with voice messages

## Architecture

```
Voice Message → STT (faster-whisper) → Claude API → TTS (Edge TTS) → Voice Reply
```

## Requirements

- Python 3.11+
- Mac Mini M4 (Apple Silicon) or compatible system
- Telegram account (phone number required)
- API keys: Telegram, Anthropic (Claude)

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

# Logging
LOG_LEVEL=INFO
```

## Usage

### First Run

On first run, you'll need to authenticate with Telegram:

```bash
python main.py
```

You'll receive a code via Telegram. Enter it when prompted. A session file will be created for future runs.

### Running the Bot

```bash
python main.py
```

The bot will start and listen for voice messages. Only the authorized user (specified by `AUTHORIZED_USER_ID`) can interact with it.

### Interacting with the Bot

1. Open Telegram and find your own chat (Saved Messages) or any private chat
2. Send a voice message
3. The bot will:
   - Transcribe your message
   - Generate a response using Claude
   - Convert it to speech
   - Reply with a voice message

## Security

- **Hardcoded authorization**: Only the user ID specified in `AUTHORIZED_USER_ID` can interact with the bot
- All other users are silently ignored
- No authentication bypass possible

## Project Structure

```
telegram-voice-call/
├── main.py                 # Entry point
├── config.py               # Configuration management
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (gitignored)
├── .env.example            # Environment template
├── bot/
│   ├── __init__.py
│   └── userbot.py          # Pyrogram userbot with voice message handler
├── audio/
│   ├── __init__.py
│   ├── stt.py              # Speech-to-text (faster-whisper)
│   └── tts.py              # Text-to-speech (Edge TTS)
└── llm/
    ├── __init__.py
    └── claude.py           # Claude API integration
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

## Next Steps (Phase 2)

Phase 2 will implement real-time voice chat using:
- pytgcalls for group voice chat
- Streaming STT with VAD (Silero)
- Real-time Claude API streaming
- Edge TTS streaming
- Interruption handling

See `PLAN.md` for full roadmap.

## License

MIT

## Support

For issues or questions, check the logs in `bot.log` first.
