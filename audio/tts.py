"""Text-to-Speech using Edge TTS."""

import logging
import asyncio
from typing import AsyncGenerator
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

    async def _decode_mp3_to_pcm(self, mp3_data: bytes, target_sample_rate: int) -> bytes:
        """
        Decode MP3 data to raw PCM using ffmpeg.
        
        Args:
            mp3_data: Raw MP3 bytes
            target_sample_rate: Target sample rate for output PCM
            
        Returns:
            Raw PCM bytes (16-bit, mono)
        """
        # Use ffmpeg to decode MP3 from stdin to raw PCM on stdout
        cmd = [
            'ffmpeg',
            '-i', 'pipe:0',           # Read from stdin
            '-f', 's16le',            # Output format: signed 16-bit little-endian
            '-acodec', 'pcm_s16le',   # PCM codec
            '-ac', '1',               # Mono
            '-ar', str(target_sample_rate),  # Target sample rate
            '-loglevel', 'error',     # Suppress verbose output
            'pipe:1'                  # Write to stdout
        ]
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        pcm_data, stderr = await proc.communicate(input=mp3_data)
        
        if proc.returncode != 0:
            logger.error(f"ffmpeg decode failed: {stderr.decode()}")
            raise RuntimeError(f"ffmpeg decode failed: {stderr.decode()}")
        
        return pcm_data

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

            # Collect all MP3 data first (Edge TTS sends partial MP3 frames)
            mp3_buffer = bytearray()

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_buffer.extend(chunk["data"])

            if not mp3_buffer:
                logger.warning("No audio data received from TTS")
                return

            # Decode MP3 to PCM using ffmpeg
            pcm_data = await self._decode_mp3_to_pcm(bytes(mp3_buffer), config.SAMPLE_RATE_WHISPER)

            # Stream in chunks
            offset = 0
            while offset < len(pcm_data):
                yield pcm_data[offset:offset + chunk_size]
                offset += chunk_size

            logger.info("TTS streaming completed")

        except Exception as e:
            logger.error(f"TTS streaming failed: {e}")
            raise

    async def stream_audio_for_telegram(
        self, 
        text: str, 
        chunk_size: int = 480,
        interrupted: asyncio.Event = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream TTS audio as PCM chunks for Telegram voice chat (24kHz, 16-bit, mono).

        IMPORTANT: pytgcalls/ntgcalls send_frame() expects 24kHz audio for external sources.
        Using 48kHz results in chipmunk (2x speed) playback.

        Args:
            text: Text to convert to speech
            chunk_size: Size of each audio chunk in bytes (default 480 = 10ms at 24kHz mono)
            interrupted: Optional asyncio.Event to check for interruption during generation

        Yields:
            Raw PCM audio chunks (16-bit, 24kHz, mono) ready for voice chat
        """
        try:
            logger.info(f"Streaming TTS for voice chat: {text[:100]}...")

            # Create Edge TTS communicator
            communicate = edge_tts.Communicate(text, self.voice)

            # Edge TTS sends MP3 chunks that can't be decoded individually.
            # We need to collect all MP3 data first, then decode.
            mp3_buffer = bytearray()

            async for chunk in communicate.stream():
                # Check for interruption during MP3 collection
                if interrupted and interrupted.is_set():
                    logger.info("TTS interrupted during MP3 collection")
                    return
                    
                if chunk["type"] == "audio":
                    mp3_buffer.extend(chunk["data"])

            # Check again before expensive decode operation
            if interrupted and interrupted.is_set():
                logger.info("TTS interrupted before decode")
                return

            if not mp3_buffer:
                logger.warning("No audio data received from TTS")
                return

            # Decode MP3 to PCM using ffmpeg at 24kHz for voice chat playback
            # ntgcalls send_frame() expects 24kHz, not 48kHz!
            pcm_data = await self._decode_mp3_to_pcm(bytes(mp3_buffer), config.SAMPLE_RATE_EXTERNAL)
            logger.info(f"TTS audio decoded: {len(pcm_data)} bytes at {config.SAMPLE_RATE_EXTERNAL}Hz")

            # Stream in chunks (480 bytes = 10ms at 24kHz mono)
            offset = 0
            chunks_yielded = 0
            while offset < len(pcm_data):
                # Check for interruption during streaming
                if interrupted and interrupted.is_set():
                    logger.info(f"TTS interrupted after {chunks_yielded} chunks ({chunks_yielded * 10}ms)")
                    return
                    
                chunk = pcm_data[offset:offset + chunk_size]
                # Pad last chunk if needed
                if len(chunk) < chunk_size:
                    chunk = chunk + b'\x00' * (chunk_size - len(chunk))
                yield chunk
                offset += chunk_size
                chunks_yielded += 1

            logger.info(f"TTS streaming completed: {chunks_yielded} chunks ({chunks_yielded * 10}ms)")

        except asyncio.CancelledError:
            logger.info("TTS streaming cancelled")
            raise
        except Exception as e:
            logger.error(f"TTS streaming failed: {e}")
            raise
