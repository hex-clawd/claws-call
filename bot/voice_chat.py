"""Voice chat integration using pytgcalls."""

import logging
import asyncio
from collections import deque
from typing import Optional
from pytgcalls import PyTgCalls
from pytgcalls.types import Update
from pytgcalls.types.input_stream import AudioPiped, InputAudioStream
from pytgcalls.types.input_stream.quality import HighQualityAudio
import config
from audio.utils import convert_telegram_to_whisper, convert_whisper_to_telegram, generate_silence

logger = logging.getLogger(__name__)


class VoiceChat:
    """Voice chat handler using pytgcalls."""

    def __init__(self, client, pipeline):
        """
        Initialize voice chat handler.

        Args:
            client: Pyrogram client instance
            pipeline: VoicePipeline instance for processing audio
        """
        self.client = client
        self.pipeline = pipeline
        self.pytgcalls = PyTgCalls(client)

        # Audio buffers
        self.input_buffer = deque()  # Buffer for incoming audio from Telegram
        self.output_buffer = deque()  # Buffer for outgoing audio to Telegram

        # State
        self.is_connected = False
        self.is_playing = False
        self.reconnect_attempts = 0

        # Configuration
        self.group_id = config.VOICE_CHAT_GROUP_ID
        self.chunk_duration_ms = 20  # 20ms chunks (standard for VoIP)
        self.samples_per_chunk_48k = int(48000 * self.chunk_duration_ms / 1000)  # 960 samples
        self.bytes_per_chunk_48k = self.samples_per_chunk_48k * 2  # 16-bit = 2 bytes per sample

        logger.info("VoiceChat initialized")

    async def start(self):
        """Start the voice chat handler and connect to pytgcalls."""
        try:
            logger.info("Starting pytgcalls...")
            await self.pytgcalls.start()

            # Register handlers
            @self.pytgcalls.on_stream_end()
            async def on_stream_end(client, update: Update):
                logger.info("Stream ended")
                await self.handle_stream_end()

            @self.pytgcalls.on_kicked()
            async def on_kicked(client, chat_id: int):
                logger.warning(f"Kicked from voice chat in {chat_id}")
                await self.handle_disconnect()

            @self.pytgcalls.on_left()
            async def on_left(client, chat_id: int):
                logger.info(f"Left voice chat in {chat_id}")
                await self.handle_disconnect()

            logger.info("pytgcalls started successfully")

        except Exception as e:
            logger.error(f"Failed to start pytgcalls: {e}")
            raise

    async def join_voice_chat(self):
        """Join the configured voice chat group."""
        try:
            if self.is_connected:
                logger.warning("Already connected to voice chat")
                return

            logger.info(f"Joining voice chat in group {self.group_id}...")

            # Create audio stream with piped input
            # This allows us to feed audio dynamically
            stream = InputAudioStream(
                AudioPiped(self._audio_producer, HighQualityAudio()),
            )

            await self.pytgcalls.join_group_call(
                self.group_id,
                stream,
                stream_type='input'
            )

            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info("Successfully joined voice chat")

            # Start audio processing loop
            asyncio.create_task(self._audio_input_loop())

        except Exception as e:
            logger.error(f"Failed to join voice chat: {e}")
            raise

    async def _audio_producer(self):
        """
        Audio producer for pytgcalls (feeds audio to the call).
        This is called by pytgcalls to get audio data to play.

        Yields:
            PCM audio bytes (16-bit, 48kHz, mono)
        """
        logger.info("Audio producer started")

        try:
            while self.is_connected:
                if self.output_buffer:
                    # Send audio from output buffer
                    chunk = self.output_buffer.popleft()
                    yield chunk
                else:
                    # Send silence when no audio is ready
                    silence = generate_silence(
                        self.chunk_duration_ms,
                        sample_rate=config.SAMPLE_RATE_TG,
                        sample_width=2
                    )
                    yield silence

                # Small delay to control chunk rate
                await asyncio.sleep(self.chunk_duration_ms / 1000.0)

        except Exception as e:
            logger.error(f"Audio producer error: {e}")
        finally:
            logger.info("Audio producer stopped")

    async def _audio_input_loop(self):
        """
        Audio input loop for receiving audio from the call.
        Note: pytgcalls doesn't have a built-in callback for receiving audio,
        so we need to use a different approach or the underlying tgcalls library.

        For now, this is a placeholder. We'll need to use the lower-level
        tgcalls API or GroupCallRaw for bidirectional audio.
        """
        # TODO: Implement audio input using lower-level API
        # This requires using py-tgcalls directly with GroupCallRaw
        logger.warning("Audio input loop not yet implemented - requires GroupCallRaw")
        pass

    async def leave_voice_chat(self):
        """Leave the voice chat."""
        try:
            if not self.is_connected:
                logger.warning("Not connected to voice chat")
                return

            logger.info("Leaving voice chat...")

            await self.pytgcalls.leave_group_call(self.group_id)

            self.is_connected = False
            self.output_buffer.clear()

            logger.info("Left voice chat successfully")

        except Exception as e:
            logger.error(f"Failed to leave voice chat: {e}")
            raise

    async def handle_stream_end(self):
        """Handle stream end event."""
        logger.info("Handling stream end...")
        self.is_playing = False

    async def handle_disconnect(self):
        """Handle disconnection from voice chat."""
        logger.warning("Handling voice chat disconnect...")
        self.is_connected = False

        # Attempt to reconnect with exponential backoff
        await self._attempt_reconnect()

    async def _attempt_reconnect(self):
        """Attempt to reconnect to voice chat with exponential backoff."""
        if self.reconnect_attempts >= 10:
            logger.error("Max reconnection attempts reached, giving up")
            return

        # Calculate backoff delay
        delay = min(
            config.AUTO_REJOIN_BACKOFF_BASE ** self.reconnect_attempts,
            config.AUTO_REJOIN_MAX_DELAY
        )
        self.reconnect_attempts += 1

        logger.info(f"Reconnecting in {delay}s (attempt {self.reconnect_attempts})...")
        await asyncio.sleep(delay)

        try:
            await self.join_voice_chat()
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            # Try again
            await self._attempt_reconnect()

    async def play_audio(self, audio_chunk: bytes):
        """
        Queue audio chunk for playback.

        Args:
            audio_chunk: PCM audio bytes (16-bit, 48kHz, mono)
        """
        self.output_buffer.append(audio_chunk)
        self.is_playing = True

    def clear_output_buffer(self):
        """Clear the output audio buffer (for interruption handling)."""
        logger.info("Clearing output audio buffer")
        self.output_buffer.clear()
        self.is_playing = False

    async def stop(self):
        """Stop the voice chat handler."""
        try:
            if self.is_connected:
                await self.leave_voice_chat()

            logger.info("Stopping pytgcalls...")
            # pytgcalls doesn't have a stop method, just leave all calls

        except Exception as e:
            logger.error(f"Error stopping voice chat: {e}")


class VoiceChatRaw:
    """
    Voice chat handler using lower-level GroupCallRaw for bidirectional audio.
    This is the correct implementation that provides both input and output callbacks.
    """

    def __init__(self, client, pipeline):
        """
        Initialize voice chat handler with GroupCallRaw.

        Args:
            client: Pyrogram client instance
            pipeline: VoicePipeline instance for processing audio
        """
        # Note: This requires py-tgcalls which provides GroupCallRaw
        # We'll implement this using the correct import path
        try:
            from pytgcalls import GroupCallFactory
            self.client = client
            self.pipeline = pipeline

            # Create GroupCallRaw instance
            self.group_call = GroupCallFactory(client).get_raw_group_call(
                on_played_data=self._on_played_data,
                on_recorded_data=self._on_recorded_data
            )

            # Audio buffers
            self.output_buffer = deque()  # Audio to play to the call

            # State
            self.is_connected = False
            self.is_playing = False
            self.reconnect_attempts = 0

            # Configuration
            self.group_id = config.VOICE_CHAT_GROUP_ID

            logger.info("VoiceChatRaw initialized with GroupCallRaw")

        except ImportError as e:
            logger.error(f"Failed to import GroupCallFactory: {e}")
            logger.error("Make sure py-tgcalls is installed correctly")
            raise

    def _on_recorded_data(self, group_call, data: bytes, length: int):
        """
        Callback when audio is received from the call.

        Args:
            group_call: GroupCall instance
            data: PCM audio bytes (16-bit, 48kHz, mono)
            length: Length of data in bytes
        """
        try:
            # Convert Telegram audio (48kHz) to Whisper format (16kHz)
            audio_16k = convert_telegram_to_whisper(data[:length])

            # Send to pipeline for processing (VAD -> STT -> LLM -> TTS)
            asyncio.create_task(self.pipeline.process_input_audio(audio_16k))

        except Exception as e:
            logger.error(f"Error processing recorded data: {e}")

    def _on_played_data(self, group_call, length: int) -> bytes:
        """
        Callback when audio is needed for playback.

        Args:
            group_call: GroupCall instance
            length: Number of bytes needed

        Returns:
            PCM audio bytes (16-bit, 48kHz, mono)
        """
        try:
            if self.output_buffer:
                # Return audio from buffer
                chunk = self.output_buffer.popleft()

                # Pad or truncate to requested length
                if len(chunk) < length:
                    chunk += generate_silence(
                        duration_ms=int((length - len(chunk)) / 96),  # 48kHz 16-bit = 96 bytes/ms
                        sample_rate=config.SAMPLE_RATE_TG
                    )
                elif len(chunk) > length:
                    chunk = chunk[:length]

                return chunk
            else:
                # Return silence
                return generate_silence(
                    duration_ms=int(length / 96),
                    sample_rate=config.SAMPLE_RATE_TG
                )

        except Exception as e:
            logger.error(f"Error generating played data: {e}")
            return b'\x00' * length

    async def join_voice_chat(self):
        """Join the voice chat."""
        try:
            if self.is_connected:
                logger.warning("Already connected to voice chat")
                return

            logger.info(f"Joining voice chat in group {self.group_id}...")

            await self.group_call.start(self.group_id)

            self.is_connected = True
            self.reconnect_attempts = 0
            logger.info("Successfully joined voice chat with GroupCallRaw")

        except Exception as e:
            logger.error(f"Failed to join voice chat: {e}")
            await self._attempt_reconnect()

    async def leave_voice_chat(self):
        """Leave the voice chat."""
        try:
            if not self.is_connected:
                return

            logger.info("Leaving voice chat...")

            await self.group_call.stop()

            self.is_connected = False
            self.output_buffer.clear()
            logger.info("Left voice chat")

        except Exception as e:
            logger.error(f"Failed to leave voice chat: {e}")

    async def play_audio(self, audio_chunk: bytes):
        """
        Queue audio for playback (48kHz PCM).

        Args:
            audio_chunk: PCM audio bytes (16-bit, 48kHz, mono)
        """
        self.output_buffer.append(audio_chunk)
        self.is_playing = True

    def clear_output_buffer(self):
        """Clear output buffer (interruption)."""
        logger.info("Clearing output buffer")
        self.output_buffer.clear()
        self.is_playing = False

    async def _attempt_reconnect(self):
        """Reconnect with exponential backoff."""
        if self.reconnect_attempts >= 10:
            logger.error("Max reconnect attempts reached")
            return

        delay = min(
            config.AUTO_REJOIN_BACKOFF_BASE ** self.reconnect_attempts,
            config.AUTO_REJOIN_MAX_DELAY
        )
        self.reconnect_attempts += 1

        logger.info(f"Reconnecting in {delay}s (attempt {self.reconnect_attempts})...")
        await asyncio.sleep(delay)

        try:
            await self.join_voice_chat()
        except Exception as e:
            logger.error(f"Reconnect failed: {e}")
            await self._attempt_reconnect()

    async def stop(self):
        """Stop voice chat handler."""
        await self.leave_voice_chat()
