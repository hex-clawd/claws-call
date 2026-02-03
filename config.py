"""Configuration management for the Telegram Voice Call bot."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Telegram userbot credentials
API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "")
PHONE_NUMBER = os.getenv("PHONE_NUMBER", "")

# Security
AUTHORIZED_USER_ID = int(os.getenv("AUTHORIZED_USER_ID", "0"))

# Claude API (not used when routing through Clawdbot gateway)
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

# Edge TTS
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "en-GB-RyanNeural")
EDGE_TTS_LANGUAGE = os.getenv("EDGE_TTS_LANGUAGE", "en-GB")

# Whisper STT
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")

# Audio settings
SAMPLE_RATE_TG = 48000  # Telegram voice message sample rate (incoming audio)
SAMPLE_RATE_WHISPER = 16000  # Whisper expects 16kHz
SAMPLE_RATE_EXTERNAL = 24000  # External source playback rate (ntgcalls expects 24kHz for send_frame)

# Voice chat settings
VOICE_CHAT_GROUP_ID = int(os.getenv("VOICE_CHAT_GROUP_ID", "0"))  # Private group ID for voice chat
VAD_SILENCE_THRESHOLD_MS = int(os.getenv("VAD_SILENCE_THRESHOLD_MS", "1500"))  # Silence duration to detect turn end
VAD_SPEECH_THRESHOLD = float(os.getenv("VAD_SPEECH_THRESHOLD", "0.5"))  # Silero VAD probability threshold
VAD_MIN_SPEECH_DURATION_MS = int(os.getenv("VAD_MIN_SPEECH_DURATION_MS", "300"))  # Min speech to be valid
AUTO_REJOIN_BACKOFF_BASE = float(os.getenv("AUTO_REJOIN_BACKOFF_BASE", "2.0"))  # Exponential backoff base
AUTO_REJOIN_MAX_DELAY = int(os.getenv("AUTO_REJOIN_MAX_DELAY", "60"))  # Max rejoin delay in seconds

# Bot mode
BOT_MODE = os.getenv("BOT_MODE", "voice_message")  # "voice_message" or "voice_chat"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Validate critical settings
def validate_config():
    """Validate that all critical configuration values are set."""
    errors = []

    if API_ID == 0:
        errors.append("API_ID not set")
    if not API_HASH:
        errors.append("API_HASH not set")
    if not PHONE_NUMBER:
        errors.append("PHONE_NUMBER not set")
    if AUTHORIZED_USER_ID == 0:
        errors.append("AUTHORIZED_USER_ID not set")
    if BOT_MODE == "voice_chat" and VOICE_CHAT_GROUP_ID == 0:
        errors.append("VOICE_CHAT_GROUP_ID not set (required for voice_chat mode)")

    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
