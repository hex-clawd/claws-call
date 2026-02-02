"""Telegram userbot implementation using Pyrogram."""

import logging
from pathlib import Path
from pyrogram import Client, filters
from pyrogram.types import Message
import config
from audio.stt import STT
from audio.tts import TTS
from llm.claude import ClaudeClient

logger = logging.getLogger(__name__)


class VoiceBot:
    """Telegram userbot that processes voice messages."""

    def __init__(self):
        """Initialize the userbot and components."""
        self.app = Client(
            "userbot",
            api_id=config.API_ID,
            api_hash=config.API_HASH,
            phone_number=config.PHONE_NUMBER
        )

        # Initialize components
        self.stt = STT()
        self.tts = TTS()
        self.claude = ClaudeClient()

        # Register handlers
        self.app.on_message(
            filters.voice & filters.private & self._is_authorized
        )(self.handle_voice_message)

        logger.info("VoiceBot initialized")

    def _is_authorized(self, _, __, message: Message) -> bool:
        """
        Security filter: only process messages from authorized user.

        Args:
            message: The incoming message

        Returns:
            True if user is authorized, False otherwise
        """
        is_auth = message.from_user.id == config.AUTHORIZED_USER_ID
        if not is_auth:
            logger.warning(f"Unauthorized access attempt from user {message.from_user.id}")
        return is_auth

    async def handle_voice_message(self, client: Client, message: Message):
        """
        Handle incoming voice messages.

        Args:
            client: Pyrogram client
            message: Voice message from user
        """
        try:
            logger.info(f"Received voice message from user {message.from_user.id}")

            # Send typing indicator
            await client.send_chat_action(message.chat.id, "typing")

            # Download voice message
            voice_file = await message.download()
            logger.info(f"Voice message downloaded: {voice_file}")

            # Transcribe audio
            await client.send_chat_action(message.chat.id, "typing")
            transcription = self.stt.transcribe(voice_file)
            logger.info(f"Transcription: {transcription}")

            if not transcription:
                await message.reply_text("Sorry, I couldn't understand that.")
                return

            # Get response from Claude
            await client.send_chat_action(message.chat.id, "typing")
            response_text = self.claude.get_response(transcription)
            logger.info(f"Claude response: {response_text}")

            # Generate TTS
            await client.send_chat_action(message.chat.id, "record_audio")
            tts_output = Path("temp_response.mp3")
            self.tts.generate(response_text, str(tts_output))

            # Send voice message reply
            await client.send_chat_action(message.chat.id, "upload_audio")
            await message.reply_voice(
                voice=str(tts_output),
                caption=response_text
            )

            # Cleanup
            Path(voice_file).unlink(missing_ok=True)
            tts_output.unlink(missing_ok=True)

            logger.info("Voice message processed successfully")

        except Exception as e:
            logger.error(f"Error processing voice message: {e}", exc_info=True)
            try:
                await message.reply_text(f"Sorry, an error occurred: {str(e)}")
            except Exception:
                pass

    def run(self):
        """Start the userbot."""
        logger.info("Starting VoiceBot...")
        self.app.run()
