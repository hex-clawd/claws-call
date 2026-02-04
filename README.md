# Telegram Voice Chat AI Assistant

Real-time voice conversation in Telegram group voice chats. The bot monitors a configured group and automatically joins when a voice chat starts, enabling live bidirectional AI conversation.

## Features

- **Live voice conversation** in Telegram group voice chats
- **Voice Activity Detection** (Silero VAD) for natural turn-taking
- **Instant interruption** - speak to interrupt the AI mid-response
- **Streaming TTS** for low-latency responses
- **Auto-rejoin** on disconnect with exponential backoff
- **OpenClaw integration** - conversations go through your OpenClaw agent

## Architecture

```
Telegram Voice Chat (pytgcalls)
    ↓ PCM 48kHz stereo
Audio Pipeline
    ↓ Resample to 16kHz mono
Silero VAD (turn detection)
    ↓ On turn end
faster-whisper STT (local)
    ↓ Transcription
OpenClaw Gateway → Your AI Agent
    ↓ Response text (streaming)
Edge TTS (sentence-level streaming)
    ↓ PCM 24kHz mono
Back to Voice Chat
```

## Requirements
## Important: This is a Userbot

This project uses **MTProto/Pyrogram** (userbot), not the Telegram Bot API. Why?

- Telegram's Bot API **doesn't support voice calls**
- Only real Telegram accounts can join voice chats
- The bot logs into a Telegram account and acts as that user

### Implications

- You need a **real Telegram account** (phone number required)
- The bot **is** that account while running — actions appear as coming from it
- Consider using a **dedicated phone number** for the bot:
  - Keeps your main account separate
  - No interference with personal Telegram use
  - Cheap prepaid SIM or eSIM works fine

### Recommendation

Get a separate SIM card or eSIM just for the bot. This way:
- Your main Telegram account stays untouched
- The bot has its own identity
- You can run it 24/7 without affecting your personal use


- Python 3.11+
- macOS (Apple Silicon) or Linux
- Telegram account (phone number required)
- [OpenClaw](https://github.com/openclaw/openclaw) running locally
- A private Telegram group for voice chat

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

Message [@userinfobot](https://t.me/userinfobot) on Telegram and copy your user ID.

### 4. Create Voice Chat Group

1. Create a new private group in Telegram
2. Add only yourself (the bot uses your account as a userbot)
3. Get the group chat ID (you can use [@RawDataBot](https://t.me/RawDataBot) or the script below)

```python
# Get group chat ID
from pyrogram import Client
app = Client("userbot", api_id=API_ID, api_hash=API_HASH)
app.start()
for dialog in app.get_dialogs():
    if dialog.chat.title:
        print(f"{dialog.chat.title}: {dialog.chat.id}")
```

### 5. Get OpenClaw Gateway Token

Your OpenClaw gateway token is in `~/.openclaw/openclaw.json` under `gateway.auth.token`:

```bash
cat ~/.openclaw/openclaw.json | grep -A2 '"auth"' | grep token
```

### 6. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# Telegram Userbot
API_ID=12345678
API_HASH=abcdef1234567890abcdef1234567890
PHONE_NUMBER=+1234567890

# Security
AUTHORIZED_USER_ID=123456789

# Voice Chat Group (negative number)
VOICE_CHAT_GROUP_ID=-1001234567890

# OpenClaw Gateway
OPENCLAW_GATEWAY_URL=ws://127.0.0.1:18789
OPENCLAW_GATEWAY_TOKEN=your_token_here

# Whisper model (tiny.en, base.en, small.en, medium.en)
WHISPER_MODEL=small.en
```

## Usage

### First Run

On first run, authenticate with Telegram:

```bash
python main.py
```

Enter the code sent to your Telegram account. A session file is created for future runs.

### Normal Operation

1. Ensure OpenClaw is running (`openclaw gateway status`)
2. Start the bot: `python main.py`
3. The bot monitors the configured group
4. Start a voice chat in the group - the bot joins automatically
5. Speak naturally - the bot detects when you finish (configurable silence threshold)
6. Interrupt anytime by speaking while the bot is talking
7. End the voice chat in Telegram - the bot leaves and returns to monitoring

### Tuning VAD

Adjust `VAD_SILENCE_THRESHOLD_MS` in `.env`:
- **Lower (500ms)** = More responsive, may cut off mid-sentence
- **Higher (1000ms+)** = More patient, slower response
- **Default (700ms)** works well for most conversations

## Project Structure

```
telegram-voice-call/
├── main.py                 # Entry point
├── config.py               # Configuration
├── requirements.txt
├── bot/
│   └── voice_chat.py       # pytgcalls voice chat handler
├── audio/
│   ├── stt.py              # Speech-to-text (faster-whisper)
│   ├── tts.py              # Text-to-speech (Edge TTS)
│   ├── vad.py              # Voice Activity Detection (Silero)
│   └── utils.py            # Audio conversion utilities
├── llm/
│   └── openclaw.py         # OpenClaw Gateway client
└── pipeline/
    └── voice_pipeline.py   # Main async processing pipeline
```

## Security

- Only the configured `AUTHORIZED_USER_ID` can interact with the bot
- Gateway runs on localhost only - OpenClaw must be on the same machine

## Troubleshooting

### "VOICE_CHAT_GROUP_ID not set"
Get your group chat ID using [@RawDataBot](https://t.me/RawDataBot) or the script above. It should be a negative number like `-1001234567890`.

### Bot doesn't join voice chat
- Check that you're in the correct group
- Verify the group ID in `.env`
- Check `bot.log` for errors

### "Gateway connection failed"
- Ensure OpenClaw is running: `openclaw gateway status`
- Verify your gateway token in `.env`

### Audio issues
- Check that ffmpeg is installed: `ffmpeg -version`
- Try a smaller Whisper model: `WHISPER_MODEL=base.en`

## License

MIT
