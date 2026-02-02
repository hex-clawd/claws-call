"""Text-to-Speech using Edge TTS."""

import logging
import asyncio
from pathlib import Path
from typing import AsyncGenerator
import edge_tts
import numpy as np
import soundfile as sf
import io
import config
from audio.utils import resample_audio, pcm_bytes_to_float32, float32_to_pcm_bytes

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

    async def stream_audio_chunks(self, text: str, chunk_size: int = 4096) -> AsyncGenerator[bytes, None]:
        """
        Stream TTS audio as raw PCM chunks (16kHz, 16-bit, mono).

        Args:
            text: Text to convert to speech
            chunk_size: Size of each audio chunk in bytes

        Yields:
            Raw PCM audio chunks (16-bit, 16kHz, mono)
        """
        try:
            logger.info(f"Streaming TTS for text: {text[:100]}...")

            # Create Edge TTS communicator
            communicate = edge_tts.Communicate(text, self.voice)

            # Buffer to accumulate audio
            audio_buffer = bytearray()

            # Stream audio chunks from Edge TTS
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    # Edge TTS returns MP3 audio data
                    audio_data = chunk["data"]

                    # Decode MP3 to PCM using soundfile
                    # Edge TTS uses 24kHz by default
                    try:
                        audio_float, sr = sf.read(io.BytesIO(audio_data))

                        # Resample to 16kHz if needed
                        if sr != config.SAMPLE_RATE_WHISPER:
                            audio_float = resample_audio(
                                audio_float,
                                orig_sr=sr,
                                target_sr=config.SAMPLE_RATE_WHISPER
                            )

                        # Convert to PCM bytes
                        pcm_chunk = float32_to_pcm_bytes(audio_float, sample_width=2)
                        audio_buffer.extend(pcm_chunk)

                        # Yield complete chunks
                        while len(audio_buffer) >= chunk_size:
                            yield bytes(audio_buffer[:chunk_size])
                            audio_buffer = audio_buffer[chunk_size:]

                    except Exception as e:
                        logger.error(f"Failed to decode MP3 chunk: {e}")
                        continue

            # Yield remaining audio
            if audio_buffer:
                yield bytes(audio_buffer)

            logger.info("TTS streaming completed")

        except Exception as e:
            logger.error(f"TTS streaming failed: {e}")
            raise

    async def stream_audio_for_telegram(self, text: str, chunk_size: int = 3840) -> AsyncGenerator[bytes, None]:
        """
        Stream TTS audio as PCM chunks for Telegram (48kHz, 16-bit, mono).

        Args:
            text: Text to convert to speech
            chunk_size: Size of each audio chunk in bytes (default 3840 = 20ms at 48kHz)

        Yields:
            Raw PCM audio chunks (16-bit, 48kHz, mono) ready for Telegram
        """
        try:
            logger.info(f"Streaming TTS for Telegram: {text[:100]}...")

            # Create Edge TTS communicator
            communicate = edge_tts.Communicate(text, self.voice)

            # Buffer to accumulate audio
            audio_buffer = bytearray()

            # Stream audio chunks from Edge TTS
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    # Edge TTS returns MP3 audio data
                    audio_data = chunk["data"]

                    # Decode MP3 to PCM using soundfile
                    try:
                        audio_float, sr = sf.read(io.BytesIO(audio_data))

                        # Resample to 48kHz for Telegram
                        if sr != config.SAMPLE_RATE_TG:
                            audio_float = resample_audio(
                                audio_float,
                                orig_sr=sr,
                                target_sr=config.SAMPLE_RATE_TG
                            )

                        # Convert to PCM bytes
                        pcm_chunk = float32_to_pcm_bytes(audio_float, sample_width=2)
                        audio_buffer.extend(pcm_chunk)

                        # Yield complete chunks
                        while len(audio_buffer) >= chunk_size:
                            yield bytes(audio_buffer[:chunk_size])
                            audio_buffer = audio_buffer[chunk_size:]

                    except Exception as e:
                        logger.error(f"Failed to decode MP3 chunk: {e}")
                        continue

            # Yield remaining audio
            if audio_buffer:
                yield bytes(audio_buffer)

            logger.info("TTS streaming for Telegram completed")

        except Exception as e:
            logger.error(f"TTS streaming failed: {e}")
            raise
