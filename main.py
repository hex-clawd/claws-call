#!/usr/bin/env python3
"""
Telegram Voice Call AI Assistant

Entry point for the voice assistant bot.
Supports both Phase 1 (voice messages) and Phase 2 (real-time voice chat).
"""

import logging
import sys
import asyncio
from bot.userbot import VoiceBot
import config


def setup_logging():
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('bot.log')
        ]
    )


async def run_voice_chat_mode():
    """Run in voice chat mode (Phase 2)."""
    from pyrogram import Client
    from bot.voice_chat import VoiceChatRaw
    from pipeline.voice_pipeline import VoicePipeline

    logger = logging.getLogger(__name__)

    # Create Pyrogram client
    app = Client(
        "userbot",
        api_id=config.API_ID,
        api_hash=config.API_HASH,
        phone_number=config.PHONE_NUMBER
    )

    try:
        await app.start()
        logger.info("Pyrogram client started")

        # Create pipeline first (will be referenced by voice chat)
        # We'll create a placeholder and set it later
        voice_chat = None
        pipeline = None

        # Create voice chat handler
        voice_chat = VoiceChatRaw(app, None)  # Pipeline will be set later

        # Create pipeline with voice chat handler
        pipeline = VoicePipeline(voice_chat)
        voice_chat.pipeline = pipeline  # Set the pipeline reference

        # Start pipeline
        await pipeline.start()

        # Join voice chat
        await voice_chat.join_voice_chat()

        logger.info("Voice chat mode active. Speak to the bot in the voice chat!")

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Stopping voice chat mode...")
        if voice_chat:
            await voice_chat.stop()
        if pipeline:
            await pipeline.stop()
        await app.stop()
    except Exception as e:
        logger.error(f"Error in voice chat mode: {e}", exc_info=True)
        if voice_chat:
            await voice_chat.stop()
        if pipeline:
            await pipeline.stop()
        await app.stop()
        raise


def run_voice_message_mode():
    """Run in voice message mode (Phase 1)."""
    logger = logging.getLogger(__name__)
    bot = VoiceBot()
    logger.info("=" * 60)
    logger.info("Telegram Voice Call AI Assistant - Voice Message Mode")
    logger.info("=" * 60)
    logger.info(f"Authorized user ID: {config.AUTHORIZED_USER_ID}")
    logger.info(f"Whisper model: {config.WHISPER_MODEL}")
    logger.info(f"Claude model: {config.CLAUDE_MODEL}")
    logger.info("Bot is ready. Send a voice message to start!")
    logger.info("=" * 60)
    bot.run()


def main():
    """Main entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Validate configuration
        config.validate_config()
        logger.info("Configuration validated successfully")

        # Determine mode
        mode = config.BOT_MODE

        logger.info("=" * 60)
        logger.info(f"Telegram Voice Call AI Assistant - {mode.upper().replace('_', ' ')} MODE")
        logger.info("=" * 60)
        logger.info(f"Authorized user ID: {config.AUTHORIZED_USER_ID}")
        logger.info(f"Whisper model: {config.WHISPER_MODEL}")
        logger.info(f"Claude model: {config.CLAUDE_MODEL}")

        if mode == "voice_chat":
            logger.info(f"Voice chat group ID: {config.VOICE_CHAT_GROUP_ID}")
            logger.info(f"VAD silence threshold: {config.VAD_SILENCE_THRESHOLD_MS}ms")
            logger.info("=" * 60)
            logger.info("Starting voice chat mode...")
            asyncio.run(run_voice_chat_mode())
        else:
            logger.info("=" * 60)
            logger.info("Starting voice message mode...")
            run_voice_message_mode()

    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        logger.error("Please check your .env file and ensure all required variables are set.")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
