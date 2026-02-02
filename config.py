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

# Claude API
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-20241022")

# Edge TTS
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "en-GB-RyanNeural")
EDGE_TTS_LANGUAGE = os.getenv("EDGE_TTS_LANGUAGE", "en-GB")

# Whisper STT
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small.en")

# Audio settings
SAMPLE_RATE_TG = 48000  # Telegram voice message sample rate
SAMPLE_RATE_WHISPER = 16000  # Whisper expects 16kHz

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
    if not ANTHROPIC_API_KEY:
        errors.append("ANTHROPIC_API_KEY not set")

    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
