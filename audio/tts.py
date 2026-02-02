"""Text-to-Speech using ElevenLabs."""

import logging
from pathlib import Path
from elevenlabs.client import ElevenLabs
from elevenlabs import save
import config

logger = logging.getLogger(__name__)


class TTS:
    """Text-to-Speech using ElevenLabs API."""

    def __init__(self):
        """Initialize the ElevenLabs client."""
        self.client = ElevenLabs(api_key=config.ELEVENLABS_API_KEY)
        self.voice_id = config.ELEVENLABS_VOICE_ID
        logger.info(f"ElevenLabs TTS initialized with voice: {self.voice_id}")

    def generate(self, text: str, output_path: str) -> str:
        """
        Generate speech from text and save to file.

        Args:
            text: Text to convert to speech
            output_path: Path to save the audio file

        Returns:
            Path to the generated audio file
        """
        try:
            logger.info(f"Generating TTS for text: {text[:100]}...")

            # Generate audio using ElevenLabs
            audio = self.client.generate(
                text=text,
                voice=self.voice_id,
                model="eleven_turbo_v2_5"
            )

            # Save to file
            save(audio, output_path)

            logger.info(f"TTS audio saved to: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise
