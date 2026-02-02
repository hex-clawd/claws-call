#!/usr/bin/env python3
"""
Telegram Voice Call AI Assistant - Phase 1: Voice Message MVP

Entry point for the voice assistant bot.
"""

import logging
import sys
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


def main():
    """Main entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Validate configuration
        config.validate_config()
        logger.info("Configuration validated successfully")

        # Start the bot
        bot = VoiceBot()
        logger.info("=" * 60)
        logger.info("Telegram Voice Call AI Assistant - Phase 1")
        logger.info("=" * 60)
        logger.info(f"Authorized user ID: {config.AUTHORIZED_USER_ID}")
        logger.info(f"Whisper model: {config.WHISPER_MODEL}")
        logger.info(f"Claude model: {config.CLAUDE_MODEL}")
        logger.info("Bot is ready. Send a voice message to start!")
        logger.info("=" * 60)

        bot.run()

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
