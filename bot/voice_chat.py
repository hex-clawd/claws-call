"""Voice chat integration using pytgcalls 2.x API."""

import logging
import asyncio
from collections import deque
from typing import Optional
from pytgcalls import PyTgCalls, filters
from pytgcalls.types import (
    MediaStream,
    AudioQuality,
    RecordStream,
    StreamFrames,
    StreamEnded,
    Direction,
    Device,
    ExternalMedia,
)
from pytgcalls.types.raw import AudioParameters, AudioStream, Stream
from ntgcalls import MediaSource
import config
from audio.utils import convert_telegram_to_whisper, convert_whisper_to_telegram, generate_silence

logger = logging.getLogger(__name__)

# Audio parameters for external source output
# IMPORTANT: ntgcalls expects 24kHz for external audio sources via send_frame()
# Using 48kHz results in chipmunk (2x speed) playback
AUDIO_PARAMS_MONO = AudioParameters(
    bitrate=config.SAMPLE_RATE_EXTERNAL,  # 24kHz - matches what ntgcalls expects
    channels=1,  # Mono for TTS playback
)

# Audio parameters for stereo (for receiving - Telegram sends stereo at 48kHz)
AUDIO_PARAMS_STEREO = AudioParameters(
    bitrate=config.SAMPLE_RATE_TG,
    channels=2,  # Stereo for receiving
)

# Chunk size for 10ms of audio at 24kHz mono 16-bit
# 24000 samples/sec * 0.01 sec * 2 bytes/sample * 1 channel = 480 bytes
CHUNK_SIZE = config.SAMPLE_RATE_EXTERNAL * 2 * 1 // 100  # 10ms chunks (mono, 24kHz)


class VoiceChat:
    """Voice chat handler using pytgcalls 2.x API.

    This implementation uses:
    - MediaStream with ExternalMedia.AUDIO for output (sending audio to the call)
    - RecordStream with on_update for input (receiving audio from the call)
    - send_frame() for dynamic audio playback
    """

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
        self.output_buffer = deque()  # Buffer for outgoing audio to Telegram

        # State
        self.is_connected = False
        self.is_playing = False
        self.reconnect_attempts = 0
        self._playback_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None
        
        # Voice chat lifecycle flag - True if the voice chat itself is still active
        # This is different from is_connected - the chat can be active but we got disconnected
        # Only set to False when the voice chat truly ends (via on_voice_chat_ended)
        self._voice_chat_active = False
        
        # Lock for reconnection to prevent multiple concurrent reconnect attempts
        self._reconnect_lock = asyncio.Lock()

        # Configuration
        self.group_id = config.VOICE_CHAT_GROUP_ID
        self.max_reconnect_attempts = 10

        logger.info("VoiceChat initialized for pytgcalls 2.x")

    async def start(self):
        """Start the voice chat handler (deprecated - use join_voice_chat directly)."""
        # Handler registration moved to join_voice_chat() to ensure proper ordering
        logger.warning("start() is deprecated - handlers are now registered in join_voice_chat()")
        pass

    async def _handle_incoming_audio(self, update: StreamFrames):
        """Handle incoming audio frames from the call.

        Args:
            update: StreamFrames containing audio data from participants
        """
        try:
            # Log that we're receiving frames (with rate limiting to avoid spam)
            if not hasattr(self, '_frame_count'):
                self._frame_count = 0
            self._frame_count += 1
            if self._frame_count == 1:
                logger.info(f"Receiving audio frames from chat {update.chat_id}")
            elif self._frame_count % 100 == 0:  # Log every 100 frames (~1 second)
                logger.debug(f"Received {self._frame_count} audio frames so far")

            # Process each frame (each participant's audio)
            for frame in update.frames:
                # frame.frame contains PCM audio bytes (48kHz, 16-bit, stereo)
                audio_data = frame.frame

                if self._frame_count <= 5:
                    logger.info(f"Audio frame received: {len(audio_data)} bytes from participant")

                # Convert from Telegram format (48kHz stereo) to Whisper format (16kHz mono)
                audio_16k = convert_telegram_to_whisper(audio_data, stereo=True)

                if self._frame_count <= 5:
                    logger.info(f"Converted to 16kHz mono: {len(audio_16k)} samples")

                # Send to pipeline for processing (VAD -> STT -> LLM -> TTS)
                if self.pipeline:
                    asyncio.create_task(self.pipeline.process_input_audio(audio_16k))

        except Exception as e:
            logger.error(f"Error processing incoming audio: {e}")

    async def join_voice_chat(self):
        """Join the configured voice chat group.
        
        This follows the official pytgcalls bridged_calls example:
        https://github.com/pytgcalls/pytgcalls/blob/master/example/bridged_calls/
        
        Key: Must call play() FIRST with ExternalMedia, THEN record()!
        """
        try:
            if self.is_connected:
                logger.warning("Already connected to voice chat")
                return

            logger.info(f"Joining voice chat in group {self.group_id}...")

            # Step 1: Start pytgcalls
            await self.pytgcalls.start()
            logger.info("pytgcalls started")

            # Step 2: Call play() FIRST with ExternalMedia.AUDIO to join call
            # This sets up outgoing audio via send_frame()
            # IMPORTANT: Use MONO (1 channel) for external source since TTS generates mono audio
            # Using stereo here while sending mono frames causes "External source not initialized"
            external_mono_stream = Stream(
                microphone=AudioStream(
                    media_source=MediaSource.EXTERNAL,
                    path='',  # Empty path for external source
                    parameters=AUDIO_PARAMS_MONO,  # 48kHz MONO
                ),
            )
            
            await self.pytgcalls.play(
                self.group_id,
                external_mono_stream,
            )
            logger.info("Joined call with play() + ExternalMedia.AUDIO (mono)")

            # Step 3: Call record() to enable INCOMING audio frame callbacks
            # This sets up the PLAYBACK stream sources for receiving audio
            await self.pytgcalls.record(
                self.group_id,
                RecordStream(
                    True,  # audio=True
                    AudioQuality.HIGH,  # Must match play() parameters!
                ),
            )
            logger.info("RecordStream enabled - listening for incoming audio")

            # Step 4: Register handlers
            # Handler for ALL stream_frame events (debug)
            @self.pytgcalls.on_update(filters.stream_frame())
            def on_any_frame(client: PyTgCalls, update: StreamFrames):
                logger.info(f"FRAME: dir={update.direction}, dev={update.device}, count={len(update.frames)}")

            # Handler for incoming audio - EXACTLY as in bridged_calls example
            @self.pytgcalls.on_update(
                filters.stream_frame(
                    Direction.INCOMING,
                    Device.MICROPHONE,
                )
            )
            def on_incoming_audio(client: PyTgCalls, update: StreamFrames):
                logger.info(f"INCOMING AUDIO: {len(update.frames)} frames")
                asyncio.create_task(self._handle_incoming_audio(update))

            # Handler for stream end
            @self.pytgcalls.on_update(filters.stream_end)
            def on_stream_end(client: PyTgCalls, update: StreamEnded):
                logger.info(f"Stream ended for chat {update.chat_id}")
                asyncio.create_task(self.handle_stream_end())

            logger.info("Audio frame handlers registered")

            self.is_connected = True
            self._voice_chat_active = True
            self.reconnect_attempts = 0
            logger.info("Successfully joined voice chat - waiting for audio frames")

            # Start playback loop for outputting buffered audio
            self._playback_task = asyncio.create_task(self._playback_loop())

        except Exception as e:
            logger.error(f"Failed to join voice chat: {e}")
            raise

    async def _playback_loop(self):
        """Loop that sends buffered audio to the call using send_frame()."""
        logger.info("Playback loop started")
        frames_sent = 0

        try:
            while self.is_connected:
                if self.output_buffer:
                    # Get chunk from buffer
                    chunk = self.output_buffer.popleft()

                    try:
                        # Send to the call
                        await self.pytgcalls.send_frame(
                            self.group_id,
                            Device.MICROPHONE,
                            chunk,
                        )
                        frames_sent += 1
                        
                        # Log progress periodically
                        if frames_sent == 1:
                            logger.info(f"Sending first audio frame ({len(chunk)} bytes)")
                        elif frames_sent % 100 == 0:  # Log every 100 frames (~1 second)
                            logger.info(f"Sent {frames_sent} audio frames, buffer: {len(self.output_buffer)}")

                    except Exception as frame_error:
                        error_str = str(frame_error).lower()
                        # Check if this is a "not in call" or disconnect error
                        if "not in a call" in error_str or "not in call" in error_str or "not initialized" in error_str:
                            logger.warning(f"Disconnected from call (frame error): {frame_error}")
                            self.is_connected = False
                            # Trigger reconnection if voice chat is still active
                            if self._voice_chat_active:
                                asyncio.create_task(self._trigger_reconnect())
                            break
                        else:
                            # Other error - log but continue trying
                            logger.error(f"send_frame error: {frame_error}")

                    # Wait for chunk duration (10ms for 960 bytes at 48kHz mono)
                    await asyncio.sleep(0.01)
                else:
                    # No audio to play - send silence to keep connection alive
                    # 960 bytes of silence for 10ms at 48kHz 16-bit mono
                    silence = b'\x00' * 960
                    try:
                        await self.pytgcalls.send_frame(
                            self.group_id,
                            Device.MICROPHONE,
                            silence,
                        )
                    except Exception:
                        pass  # Ignore errors on silence frames
                    await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            logger.info(f"Playback loop cancelled (sent {frames_sent} frames)")
        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"Playback loop error after {frames_sent} frames: {e}")
            # Check if this is a disconnect error that warrants reconnection
            if self._voice_chat_active and ("not in a call" in error_str or "not in call" in error_str):
                self.is_connected = False
                asyncio.create_task(self._trigger_reconnect())
        finally:
            logger.info(f"Playback loop stopped (total frames sent: {frames_sent})")

    async def leave_voice_chat(self, voice_chat_ended: bool = True):
        """Leave the voice chat.
        
        Args:
            voice_chat_ended: If True, the voice chat itself has ended (don't try to reconnect).
                            If False, we're just leaving but the chat may continue.
        """
        try:
            logger.info(f"Leaving voice chat (voice_chat_ended={voice_chat_ended})...")

            # If voice chat ended, stop any reconnection attempts
            if voice_chat_ended:
                self._voice_chat_active = False
                
            # Cancel any pending reconnect task
            if self._reconnect_task and not self._reconnect_task.done():
                self._reconnect_task.cancel()
                try:
                    await self._reconnect_task
                except asyncio.CancelledError:
                    pass
                self._reconnect_task = None

            # Cancel playback task
            if self._playback_task:
                self._playback_task.cancel()
                try:
                    await self._playback_task
                except asyncio.CancelledError:
                    pass
                self._playback_task = None

            # Only try to leave the call if we think we're connected
            if self.is_connected:
                try:
                    await self.pytgcalls.leave_call(self.group_id)
                except Exception as leave_error:
                    # Might already be disconnected, that's fine
                    logger.debug(f"leave_call error (may be expected): {leave_error}")

            self.is_connected = False
            self.output_buffer.clear()
            self.reconnect_attempts = 0

            logger.info("Left voice chat successfully")

        except Exception as e:
            logger.error(f"Failed to leave voice chat: {e}")
            # Still clean up state
            self.is_connected = False
            self._voice_chat_active = False

    async def handle_stream_end(self):
        """Handle stream end event."""
        logger.info("Handling stream end...")
        self.is_playing = False

    async def handle_disconnect(self):
        """Handle disconnection from voice chat (called externally or from pytgcalls events)."""
        logger.warning("Handling voice chat disconnect...")
        self.is_connected = False

        # Cancel playback task
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
            self._playback_task = None

        # Attempt to reconnect if voice chat is still active
        if self._voice_chat_active:
            await self._trigger_reconnect()
        else:
            logger.info("Voice chat not active, skipping reconnect")

    async def _trigger_reconnect(self):
        """Trigger a reconnection attempt (called from playback loop or other error handlers)."""
        # Use lock to prevent multiple concurrent reconnect attempts
        if self._reconnect_lock.locked():
            logger.debug("Reconnect already in progress, skipping")
            return
            
        async with self._reconnect_lock:
            if not self._voice_chat_active:
                logger.info("Voice chat no longer active, not reconnecting")
                return
                
            if self.is_connected:
                logger.debug("Already connected, no need to reconnect")
                return
                
            await self._attempt_reconnect()

    async def _attempt_reconnect(self):
        """Attempt to reconnect to voice chat with exponential backoff."""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached, giving up")
            self._voice_chat_active = False
            return

        # Calculate backoff delay with exponential backoff
        delay = min(
            config.AUTO_REJOIN_BACKOFF_BASE ** self.reconnect_attempts,
            config.AUTO_REJOIN_MAX_DELAY
        )
        self.reconnect_attempts += 1

        logger.info(f"Reconnecting in {delay:.1f}s (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})...")
        await asyncio.sleep(delay)

        # Check again if we should still reconnect
        if not self._voice_chat_active:
            logger.info("Voice chat ended during reconnect delay, aborting")
            return

        try:
            # Clean up any existing state before rejoining
            await self._cleanup_before_reconnect()
            
            # Rejoin the call
            await self.join_voice_chat()
            logger.info("Successfully reconnected to voice chat!")
            
        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"Reconnection attempt {self.reconnect_attempts} failed: {e}")
            
            # If the call itself doesn't exist anymore, stop trying
            if "call not found" in error_str or "no active call" in error_str or "video chat" in error_str:
                logger.info("Voice chat appears to have ended, stopping reconnection attempts")
                self._voice_chat_active = False
                return
            
            # Try again if voice chat is still active
            if self._voice_chat_active:
                await self._attempt_reconnect()

    async def _cleanup_before_reconnect(self):
        """Clean up state before attempting to reconnect."""
        logger.debug("Cleaning up before reconnect...")
        
        # Cancel playback task if still running
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
            try:
                await self._playback_task
            except asyncio.CancelledError:
                pass
            self._playback_task = None
        
        # Clear buffers but preserve state flags
        self.output_buffer.clear()
        self.is_connected = False
        self.is_playing = False
        
        # Small delay to let things settle
        await asyncio.sleep(0.5)

    async def play_audio(self, audio_chunk: bytes):
        """
        Queue audio chunk for playback.

        Args:
            audio_chunk: PCM audio bytes (16-bit, 48kHz, mono)
        """
        # Split into appropriate chunk sizes for send_frame
        # Each chunk should be ~10ms of audio
        offset = 0
        while offset < len(audio_chunk):
            chunk = audio_chunk[offset:offset + CHUNK_SIZE]

            # Pad if necessary (last chunk might be smaller)
            if len(chunk) < CHUNK_SIZE:
                chunk = chunk + b'\x00' * (CHUNK_SIZE - len(chunk))

            self.output_buffer.append(chunk)
            offset += CHUNK_SIZE

        self.is_playing = True

    def clear_output_buffer(self):
        """Clear the output audio buffer (for interruption handling)."""
        buffer_size = len(self.output_buffer)
        logger.info(f"CLEAR BUFFER: Clearing {buffer_size} chunks from output buffer")
        self.output_buffer.clear()
        self.is_playing = False
        logger.info("CLEAR BUFFER: Output buffer cleared, is_playing=False")

    async def stop(self):
        """Stop the voice chat handler completely."""
        try:
            # Mark voice chat as no longer active to prevent reconnection
            self._voice_chat_active = False
            
            if self.is_connected:
                await self.leave_voice_chat(voice_chat_ended=True)
            else:
                # Still need to clean up tasks even if not connected
                if self._reconnect_task and not self._reconnect_task.done():
                    self._reconnect_task.cancel()
                    try:
                        await self._reconnect_task
                    except asyncio.CancelledError:
                        pass
                    
                if self._playback_task and not self._playback_task.done():
                    self._playback_task.cancel()
                    try:
                        await self._playback_task
                    except asyncio.CancelledError:
                        pass

            logger.info("Voice chat handler stopped")

        except Exception as e:
            logger.error(f"Error stopping voice chat: {e}")
