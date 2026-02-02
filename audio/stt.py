"""Speech-to-Text using faster-whisper."""

import logging
from pathlib import Path
from faster_whisper import WhisperModel
import config

logger = logging.getLogger(__name__)


class STT:
    """Speech-to-Text transcription using faster-whisper."""

    def __init__(self):
        """Initialize the Whisper model."""
        logger.info(f"Loading Whisper model: {config.WHISPER_MODEL}")

        # Use CPU for now, can add device selection later
        self.model = WhisperModel(
            config.WHISPER_MODEL,
            device="cpu",
            compute_type="int8"
        )
        logger.info("Whisper model loaded successfully")

    def transcribe(self, audio_path: str) -> str:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        try:
            logger.info(f"Transcribing audio: {audio_path}")

            # Transcribe with faster-whisper
            segments, info = self.model.transcribe(
                audio_path,
                beam_size=1,  # Greedy decoding for speed
                language="en",
                condition_on_previous_text=False
            )

            # Combine all segments
            text = " ".join([segment.text for segment in segments])
            text = text.strip()

            logger.info(f"Transcription complete: {len(text)} chars")
            return text

        except Exception as e:
            logger.error(f"STT transcription failed: {e}")
            raise
