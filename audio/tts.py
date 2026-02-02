"""Text-to-Speech using Edge TTS."""

import logging
import asyncio
from pathlib import Path
import edge_tts
import config

logger = logging.getLogger(__name__)


class TTS:
    """Text-to-Speech using Edge TTS (Microsoft Edge's free TTS service)."""

    def __init__(self):
        """Initialize the Edge TTS client."""
        self.voice = config.EDGE_TTS_VOICE
        self.language = config.EDGE_TTS_LANGUAGE
        logger.info(f"Edge TTS initialized with voice: {self.voice}")

    async def generate_async(self, text: str, output_path: str) -> str:
        """
        Generate speech from text and save to file (async version).

        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file

        Returns:
            Path to the generated audio file
        """
        try:
            logger.info(f"Generating TTS for text: {text[:100]}...")

            # Generate audio using Edge TTS
            communicate = edge_tts.Communicate(text, self.voice)
            await communicate.save(output_path)

            logger.info(f"TTS audio saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise

    def generate(self, text: str, output_path: str) -> str:
        """
        Generate speech from text and save to file (sync wrapper).

        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file

        Returns:
            Path to the generated audio file
        """
        return asyncio.run(self.generate_async(text, output_path))
