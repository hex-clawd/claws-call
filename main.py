#!/usr/bin/env python3
"""
Telegram Voice Chat AI Assistant

Real-time voice conversation in Telegram group voice chats.
Bot monitors a group and auto-joins when a voice chat starts.
"""

# Monkey-patch pyrogram.errors for pytgcalls compatibility
# pytgcalls expects GroupcallForbidden which doesn't exist in pyrofork
def _patch_pyrogram_errors():
    import pyrogram.errors
    class GroupcallForbidden(Exception):
        pass
    pyrogram.errors.GroupcallForbidden = GroupcallForbidden
_patch_pyrogram_errors()

import logging
import sys
import asyncio
from pyrogram import Client, filters
from pyrogram.raw.functions.channels import GetFullChannel
from bot.voice_chat import VoiceChat
from pipeline.voice_pipeline import VoicePipeline
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


async def run_voice_chat():
    """Run the voice chat bot.
    
    Bot monitors the group and auto-joins when a voice chat starts.
    Leaves when the voice chat ends.
    """
    logger = logging.getLogger(__name__)

    # Create Pyrogram client
    app = Client(
        "userbot",
        api_id=config.API_ID,
        api_hash=config.API_HASH,
        phone_number=config.PHONE_NUMBER
    )

    # State containers (will be initialized after app starts)
    voice_chat = None
    pipeline = None

    async def join_call():
        """Join the voice chat and start processing."""
        nonlocal voice_chat, pipeline
        
        if voice_chat and voice_chat.is_connected:
            logger.info("Already connected to voice chat, skipping join")
            return
            
        logger.info("Joining voice chat...")
        
        # Create voice chat handler
        voice_chat = VoiceChat(app, None)
        
        # Create pipeline with voice chat handler
        pipeline = VoicePipeline(voice_chat)
        voice_chat.pipeline = pipeline
        
        # Start pipeline and join
        await pipeline.start()
        await voice_chat.join_voice_chat()
        
        logger.info("Voice chat joined! Bot is now active in the call.")

    async def leave_call(voice_chat_ended: bool = True):
        """Leave the voice chat and clean up.
        
        Args:
            voice_chat_ended: If True, the voice chat itself has ended (from Telegram event).
                            If False, we're just disconnecting but chat may continue.
        """
        nonlocal voice_chat, pipeline
        
        if not voice_chat:
            logger.info("No voice_chat instance, nothing to leave")
            return
            
        # Check if we're connected OR if voice chat ended (need to clean up either way)
        if not voice_chat.is_connected and not voice_chat._voice_chat_active and not voice_chat_ended:
            logger.info("Not in voice chat and not active, nothing to leave")
            return
            
        logger.info(f"Leaving voice chat (voice_chat_ended={voice_chat_ended})...")
        
        if voice_chat:
            await voice_chat.stop()
            voice_chat = None
        if pipeline:
            await pipeline.stop()
            pipeline = None
            
        logger.info("Left voice chat. Back to monitoring mode.")

    try:
        await app.start()
        logger.info("Pyrogram client started")

        # Register handler for voice chat started
        @app.on_message(filters.chat(config.VOICE_CHAT_GROUP_ID) & filters.video_chat_started)
        async def on_voice_chat_started(client, message):
            """Called when a voice chat starts in the monitored group."""
            logger.info(f"Voice chat started in group {message.chat.id}!")
            await join_call()

        # Register handler for voice chat ended
        @app.on_message(filters.chat(config.VOICE_CHAT_GROUP_ID) & filters.video_chat_ended)
        async def on_voice_chat_ended(client, message):
            """Called when a voice chat ends in the monitored group."""
            logger.info(f"Voice chat ENDED in group {message.chat.id}!")
            await leave_call(voice_chat_ended=True)

        logger.info(f"Monitoring group {config.VOICE_CHAT_GROUP_ID} for voice chats...")
        logger.info("Bot will auto-join when a voice chat starts!")
        
        # Check if there's already an active call
        try:
            peer = await app.resolve_peer(config.VOICE_CHAT_GROUP_ID)
            full_chat = await app.invoke(GetFullChannel(channel=peer))
            if hasattr(full_chat.full_chat, 'call') and full_chat.full_chat.call:
                logger.info("Active voice chat detected! Joining...")
                await join_call()
        except Exception as e:
            logger.debug(f"Could not check for active call (normal for groups): {e}")

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Stopping...")
        await leave_call(voice_chat_ended=True)
        await app.stop()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        await leave_call(voice_chat_ended=True)
        await app.stop()
        raise


def main():
    """Main entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Validate configuration
        config.validate_config()
        logger.info("Configuration validated successfully")

        logger.info("=" * 60)
        logger.info("Telegram Voice Chat AI Assistant")
        logger.info("=" * 60)
        logger.info(f"Authorized user ID: {config.AUTHORIZED_USER_ID}")
        logger.info(f"Voice chat group ID: {config.VOICE_CHAT_GROUP_ID}")
        logger.info(f"Whisper model: {config.WHISPER_MODEL}")
        logger.info(f"VAD silence threshold: {config.VAD_SILENCE_THRESHOLD_MS}ms")
        logger.info("=" * 60)

        asyncio.run(run_voice_chat())

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
