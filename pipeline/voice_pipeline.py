"""Main async pipeline for real-time voice processing."""

import logging
import asyncio
import numpy as np
from collections import deque
from typing import Optional
from audio.vad import VAD
from audio.stt import STT
from audio.tts import TTS
from audio.utils import convert_whisper_to_telegram
from llm.clawdbot import ClawdbotClient
import config

logger = logging.getLogger(__name__)


class VoicePipeline:
    """
    Main async pipeline for real-time voice processing.

    Flow: Audio input -> VAD -> STT -> LLM -> TTS -> Audio output
    Handles interruption, buffering, and async coordination.
    """

    def __init__(self, voice_chat_handler):
        """
        Initialize the voice pipeline.

        Args:
            voice_chat_handler: VoiceChatRaw instance for audio I/O
        """
        # Components
        self.vad = VAD()
        self.stt = STT()
        self.tts = TTS()
        self.claude = ClawdbotClient()
        self.voice_chat = voice_chat_handler

        # Audio buffers
        self.input_audio_buffer = deque()  # Buffer for incoming audio chunks (16kHz float32)
        self.stt_buffer = []  # Accumulated audio for STT when turn is detected
        
        # Silero VAD requires EXACTLY 512 samples at 16kHz (32ms).
        # Buffer small incoming chunks until we have enough for VAD.
        self.vad_accumulator = []  # Accumulates audio until we have enough for VAD
        self.vad_chunk_samples = 512  # Silero VAD requires exactly 512 samples at 16kHz (32ms)

        # State
        self.is_processing = False  # Is AI currently generating a response
        self.is_speaking = False  # Is AI currently playing TTS
        self.interrupted = asyncio.Event()  # Interruption signal
        self.llm_task: Optional[asyncio.Task] = None  # Current LLM task
        self.tts_task: Optional[asyncio.Task] = None  # Current TTS task

        # Debug counters
        self._input_frame_count = 0
        self._vad_call_count = 0

        logger.info(f"VoicePipeline initialized (VAD chunk: {self.vad_chunk_samples} samples = 32ms)")

    async def start(self):
        """Start the pipeline processing loop."""
        logger.info("Starting voice pipeline...")

        # Start the main processing loop
        asyncio.create_task(self._processing_loop())

        logger.info("Voice pipeline started")

    async def _processing_loop(self):
        """Main processing loop for handling incoming audio."""
        logger.info("Processing loop started")

        try:
            while True:
                # Wait for audio in the buffer
                if not self.input_audio_buffer:
                    await asyncio.sleep(0.01)
                    continue

                # Get audio chunk and add to VAD accumulator
                audio_chunk = self.input_audio_buffer.popleft()
                self._input_frame_count += 1
                
                # Log first few frames for debugging
                if self._input_frame_count <= 3:
                    logger.info(f"Input frame #{self._input_frame_count}: {len(audio_chunk)} samples (16kHz)")
                
                self.vad_accumulator.append(audio_chunk)
                
                # Calculate total accumulated samples
                total_samples = sum(len(chunk) for chunk in self.vad_accumulator)
                
                # Only process when we have exactly 512 samples for VAD (32ms at 16kHz)
                if total_samples < self.vad_chunk_samples:
                    if self._input_frame_count <= 5:
                        logger.debug(f"Accumulating: {total_samples}/{self.vad_chunk_samples} samples")
                    continue
                
                # Concatenate accumulated audio into a proper VAD chunk
                accumulated_audio = np.concatenate(self.vad_accumulator)
                
                # Process in vad_chunk_samples (512 samples = 32ms) chunks
                offset = 0
                while offset + self.vad_chunk_samples <= len(accumulated_audio):
                    vad_chunk = accumulated_audio[offset:offset + self.vad_chunk_samples]
                    offset += self.vad_chunk_samples
                    
                    self._vad_call_count += 1
                    if self._vad_call_count <= 3:
                        logger.info(f"VAD call #{self._vad_call_count}: {len(vad_chunk)} samples")
                    
                    # Process through VAD
                    is_speech, turn_ended = self.vad.update_state(vad_chunk)

                    if is_speech:
                        # Check for interruption
                        if self.is_speaking:
                            logger.info("INTERRUPTION DETECTED: User speaking while AI is talking")
                            await self._handle_interruption()

                        # Accumulate audio for STT
                        self.stt_buffer.append(vad_chunk)

                    elif turn_ended:
                        # User finished speaking, process the turn
                        if self.stt_buffer:
                            await self._process_user_turn()
                
                # Keep remainder for next iteration
                if offset < len(accumulated_audio):
                    self.vad_accumulator = [accumulated_audio[offset:]]
                else:
                    self.vad_accumulator = []

        except asyncio.CancelledError:
            logger.info("Processing loop cancelled")
        except Exception as e:
            logger.error(f"Error in processing loop: {e}", exc_info=True)

    async def _handle_interruption(self):
        """Handle user interruption of AI speech."""
        logger.info("Handling interruption...")

        # Set interruption flag
        self.interrupted.set()

        # Clear TTS output buffer
        self.voice_chat.clear_output_buffer()
        self.is_speaking = False

        # Cancel ongoing LLM generation
        if self.llm_task and not self.llm_task.done():
            logger.info("Cancelling LLM generation")
            self.llm_task.cancel()
            try:
                await self.llm_task
            except asyncio.CancelledError:
                pass

        # Cancel ongoing TTS generation
        if self.tts_task and not self.tts_task.done():
            logger.info("Cancelling TTS generation")
            self.tts_task.cancel()
            try:
                await self.tts_task
            except asyncio.CancelledError:
                pass

        self.is_processing = False

        # Reset VAD state
        self.vad.reset()

        # Clear STT buffer and VAD accumulator
        self.stt_buffer.clear()
        self.vad_accumulator.clear()

        logger.info("Interruption handled")

    async def _process_user_turn(self):
        """Process a complete user turn (speech -> STT -> LLM -> TTS)."""
        try:
            if self.is_processing:
                logger.warning("Already processing a turn, skipping")
                return

            self.is_processing = True
            self.interrupted.clear()

            # Concatenate buffered audio
            full_audio = np.concatenate(self.stt_buffer)
            self.stt_buffer.clear()

            logger.info(f"Processing user turn: {len(full_audio)} samples ({len(full_audio)/config.SAMPLE_RATE_WHISPER:.2f}s)")

            # Transcribe audio
            transcription = await self._transcribe_audio(full_audio)

            if not transcription or len(transcription.strip()) == 0:
                logger.info("Empty transcription, ignoring")
                self.is_processing = False
                return

            logger.info(f"Transcription: {transcription}")

            # Check for interruption before proceeding
            if self.interrupted.is_set():
                logger.info("Interrupted before LLM, aborting")
                self.is_processing = False
                return

            # Get LLM response and generate TTS
            self.llm_task = asyncio.create_task(self._generate_and_speak_response(transcription))
            await self.llm_task

        except asyncio.CancelledError:
            logger.info("User turn processing cancelled")
        except Exception as e:
            logger.error(f"Error processing user turn: {e}", exc_info=True)
        finally:
            self.is_processing = False

    async def _transcribe_audio(self, audio: np.ndarray) -> str:
        """
        Transcribe audio using faster-whisper.

        Args:
            audio: Audio data as numpy array (float32, 16kHz)

        Returns:
            Transcribed text
        """
        try:
            # Save audio to temporary file for faster-whisper
            # (faster-whisper expects file path or file-like object)
            import tempfile
            import soundfile as sf

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name
                sf.write(temp_path, audio, config.SAMPLE_RATE_WHISPER)

            # Transcribe
            transcription = self.stt.transcribe(temp_path)

            # Cleanup
            import os
            os.unlink(temp_path)

            return transcription

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    async def _generate_and_speak_response(self, user_text: str):
        """
        Generate LLM response and speak it with TTS (streaming).

        Args:
            user_text: User's transcribed text
        """
        try:
            logger.info("Generating LLM response...")

            # Get response from Clawdbot (async method)
            response_text = await self.claude.get_response(user_text)

            logger.info(f"LLM response: {response_text}")

            # Check for interruption
            if self.interrupted.is_set():
                logger.info("Interrupted after LLM, not speaking")
                return

            # Generate and play TTS
            self.is_speaking = True
            await self._speak_text(response_text)
            self.is_speaking = False

            logger.info("Response complete")

        except asyncio.CancelledError:
            logger.info("LLM response generation cancelled")
            self.is_speaking = False
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            self.is_speaking = False

    async def _speak_text(self, text: str):
        """
        Generate TTS and play to voice chat (streaming).

        Args:
            text: Text to speak
        """
        try:
            logger.info("Generating and playing TTS...")

            # Stream TTS audio chunks
            async for audio_chunk in self.tts.stream_audio_for_telegram(text):
                # Check for interruption
                if self.interrupted.is_set():
                    logger.info("TTS interrupted")
                    break

                # Send to voice chat
                await self.voice_chat.play_audio(audio_chunk)

            logger.info("TTS playback complete")

        except asyncio.CancelledError:
            logger.info("TTS playback cancelled")
        except Exception as e:
            logger.error(f"Error playing TTS: {e}", exc_info=True)

    async def process_input_audio(self, audio_chunk: np.ndarray):
        """
        Process incoming audio chunk from voice chat.

        Args:
            audio_chunk: Audio data as numpy array (float32, 16kHz)
        """
        # Add to input buffer for processing loop
        self.input_audio_buffer.append(audio_chunk)

    async def stop(self):
        """Stop the pipeline."""
        logger.info("Stopping voice pipeline...")

        # Cancel any ongoing tasks
        if self.llm_task and not self.llm_task.done():
            self.llm_task.cancel()

        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()

        # Clear buffers
        self.input_audio_buffer.clear()
        self.stt_buffer.clear()
        self.vad_accumulator.clear()

        logger.info("Voice pipeline stopped")


class StreamingClawdbotClient:
    """
    Streaming version of Claude client (for future optimization).
    This allows sentence-by-sentence TTS generation.
    """

    def __init__(self, claude_client: ClawdbotClient):
        """
        Initialize streaming Claude client.

        Args:
            claude_client: Base ClawdbotClient instance
        """
        self.client = claude_client.client
        self.conversation_history = claude_client.conversation_history

    async def stream_response(self, user_message: str):
        """
        Stream Claude response sentence by sentence.

        Args:
            user_message: User's message

        Yields:
            Sentences from Claude's response
        """
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })

            # Stream response from Claude
            full_response = ""
            current_sentence = ""

            async with self.client.messages.stream(
                model=config.CLAUDE_MODEL,
                max_tokens=1024,
                system="You are a helpful voice assistant. Keep your responses concise and conversational.",
                messages=self.conversation_history
            ) as stream:
                async for text in stream.text_stream:
                    full_response += text
                    current_sentence += text

                    # Check for sentence boundaries
                    if any(punct in text for punct in ['.', '!', '?', '\n']):
                        if current_sentence.strip():
                            yield current_sentence.strip()
                            current_sentence = ""

            # Yield remaining text
            if current_sentence.strip():
                yield current_sentence.strip()

            # Add assistant message to history
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response
            })

            # Keep history manageable
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

        except Exception as e:
            logger.error(f"Streaming Claude error: {e}")
            raise
