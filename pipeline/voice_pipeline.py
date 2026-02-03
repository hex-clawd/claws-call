"""Main async pipeline for real-time voice processing."""

import logging
import asyncio
import re
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


def strip_markdown(text: str) -> str:
    """
    Strip markdown formatting from text for TTS.
    
    Converts markdown to plain readable text so TTS doesn't read
    "asterisk" or "backtick" literally.
    
    Args:
        text: Text potentially containing markdown
        
    Returns:
        Plain text suitable for speech
    """
    # Remove code blocks (```code```)
    text = re.sub(r'```[\s\S]*?```', '', text)
    
    # Remove inline code (`code`) - extract the code content
    text = re.sub(r'`([^`]+)`', r'\1', text)
    
    # Remove images ![alt](url)
    text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'\1', text)
    
    # Convert links [text](url) to just text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    
    # Remove headers (# ## ### etc) - keep the text
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    
    # Remove bold/italic: ***text***, **text**, *text*, ___text___, __text__, _text_
    # Process triple markers first, then double, then single
    text = re.sub(r'\*\*\*([^*]+)\*\*\*', r'\1', text)
    text = re.sub(r'___([^_]+)___', r'\1', text)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)
    
    # Remove strikethrough ~~text~~
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    
    # Remove blockquotes (> text)
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    
    # Remove horizontal rules (---, ***, ___)
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
    
    # Remove bullet points (* - +) but keep the text
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    
    # Remove numbered lists (1. 2. etc) but keep the text
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
    
    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    
    return text


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
        self.llm_task: Optional[asyncio.Task] = None  # Current LLM+TTS task (TTS is awaited inside)

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

                    # Log speech detection for debugging (rate limited)
                    if is_speech and not hasattr(self, '_last_speech_log_count'):
                        self._last_speech_log_count = 0
                    if is_speech:
                        self._last_speech_log_count = getattr(self, '_last_speech_log_count', 0) + 1
                        if self._last_speech_log_count <= 3 or self._last_speech_log_count % 50 == 0:
                            buffer_size = len(self.voice_chat.output_buffer)
                            logger.info(f"🎤 SPEECH DETECTED (count={self._last_speech_log_count}, is_speaking={self.is_speaking}, buffer={buffer_size})")

                    if is_speech:
                        # Check for interruption - user speaking while bot is talking
                        # CRITICAL: Check BOTH is_speaking (TTS generating) AND output_buffer (playback in progress)
                        # is_speaking becomes False when TTS generation finishes, but playback continues!
                        buffer_has_audio = len(self.voice_chat.output_buffer) > 0
                        bot_is_outputting = self.is_speaking or buffer_has_audio
                        
                        if bot_is_outputting:
                            logger.info("!" * 50)
                            logger.info("🛑 INTERRUPTION DETECTED! User spoke during bot output!")
                            logger.info(f"  is_speaking={self.is_speaking} (TTS generating)")
                            logger.info(f"  buffer_has_audio={buffer_has_audio} (playback active)")
                            logger.info(f"  output_buffer={len(self.voice_chat.output_buffer)} chunks")
                            logger.info("!" * 50)
                            await self._handle_interruption()

                        # Accumulate audio for STT (including the interrupting speech)
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
        logger.info("=" * 60)
        logger.info("🛑 INTERRUPTION HANDLER TRIGGERED 🛑")
        logger.info(f"  State: is_speaking={self.is_speaking}, is_processing={self.is_processing}")
        logger.info(f"  LLM task: {self.llm_task}, done={self.llm_task.done() if self.llm_task else 'N/A'}")
        logger.info(f"  Output buffer: {len(self.voice_chat.output_buffer)} chunks queued")

        # 1. Set interruption flag FIRST - TTS checks this during streaming
        self.interrupted.set()
        logger.info("  ✓ Interruption flag SET")

        # 2. Clear TTS output buffer IMMEDIATELY - this stops playback
        buffer_before = len(self.voice_chat.output_buffer)
        self.voice_chat.clear_output_buffer()
        logger.info(f"  ✓ Output buffer CLEARED ({buffer_before} chunks removed)")
        
        # 3. Mark as not speaking
        self.is_speaking = False
        logger.info("  ✓ is_speaking = False")

        # 4. Cancel ongoing LLM/TTS generation task
        if self.llm_task and not self.llm_task.done():
            logger.info("  → Cancelling LLM task...")
            self.llm_task.cancel()
            try:
                await self.llm_task
            except asyncio.CancelledError:
                logger.info("  ✓ LLM task CANCELLED")
        else:
            logger.info("  ✓ No active LLM task to cancel")

        # 5. Reset processing state
        self.is_processing = False
        logger.info("  ✓ is_processing = False")

        # 6. Reset VAD state to start fresh
        self.vad.reset()
        logger.info("  ✓ VAD state RESET")

        # 7. Clear speech buffers - new speech will be accumulated fresh
        self.stt_buffer.clear()
        self.vad_accumulator.clear()
        logger.info("  ✓ STT buffer and VAD accumulator CLEARED")

        # Reset speech detection counter
        self._last_speech_log_count = 0

        logger.info("🛑 INTERRUPTION COMPLETE - Ready for new input 🛑")
        logger.info("=" * 60)

    async def _process_user_turn(self):
        """Process a complete user turn (speech -> STT -> LLM -> TTS)."""
        try:
            if self.is_processing:
                logger.warning("Already processing a turn, skipping")
                return

            self.is_processing = True
            self.interrupted.clear()  # Clear any previous interruption
            logger.info("-" * 40)
            logger.info("NEW TURN: Starting user turn processing")

            # Concatenate buffered audio
            full_audio = np.concatenate(self.stt_buffer)
            self.stt_buffer.clear()

            duration_sec = len(full_audio) / config.SAMPLE_RATE_WHISPER
            logger.info(f"NEW TURN: Audio collected: {len(full_audio)} samples ({duration_sec:.2f}s)")

            # Transcribe audio
            transcription = await self._transcribe_audio(full_audio)

            if not transcription or len(transcription.strip()) == 0:
                logger.info("NEW TURN: Empty transcription, ignoring")
                self.is_processing = False
                return

            logger.info(f"NEW TURN: Transcription: '{transcription}'")

            # Check for interruption before proceeding
            if self.interrupted.is_set():
                logger.info("NEW TURN: Interrupted before LLM, aborting")
                self.is_processing = False
                return

            # Get LLM response and generate TTS
            logger.info("NEW TURN: Starting LLM+TTS task")
            self.llm_task = asyncio.create_task(self._generate_and_speak_response(transcription))
            await self.llm_task
            logger.info("NEW TURN: Complete")

        except asyncio.CancelledError:
            logger.info("NEW TURN: Cancelled")
        except Exception as e:
            logger.error(f"NEW TURN: Error: {e}", exc_info=True)
        finally:
            self.is_processing = False
            logger.info("-" * 40)

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
            logger.info(f"LLM request: '{user_text}'")

            # Get response from Clawdbot (async method)
            response_text = await self.claude.get_response(user_text)

            logger.info(f"LLM response ({len(response_text)} chars): {response_text[:100]}...")

            # Check for interruption before TTS
            if self.interrupted.is_set():
                logger.info("INTERRUPTED: After LLM, before TTS - aborting")
                return

            # Strip markdown for clean TTS output
            clean_text = strip_markdown(response_text)
            if clean_text != response_text:
                logger.debug(f"Markdown stripped: {len(response_text)} -> {len(clean_text)} chars")

            # Generate and play TTS
            # NOTE: is_speaking=True during TTS GENERATION, but playback continues after!
            # Interruption detection also checks output_buffer for active playback
            logger.info("🔊 TTS GENERATION STARTING (is_speaking=True)")
            self.is_speaking = True
            await self._speak_text(clean_text)
            # Keep is_speaking True while buffer is draining
            logger.info(f"🔊 TTS GENERATION DONE, buffer has {len(self.voice_chat.output_buffer)} chunks remaining")
            # Wait briefly for buffer to drain, but set is_speaking=False
            # Interruption check will still work because it also checks buffer
            self.is_speaking = False
            logger.info("🔊 TTS generation finished (is_speaking=False, buffer may still be playing)")

        except asyncio.CancelledError:
            logger.info("CANCELLED: LLM/TTS task was cancelled")
            self.is_speaking = False
            raise  # Re-raise to propagate
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
            logger.info(f"TTS starting: '{text[:50]}...' ({len(text)} chars)")
            chunks_played = 0

            # Stream TTS audio chunks - pass interrupted event for early exit
            async for audio_chunk in self.tts.stream_audio_for_telegram(text, interrupted=self.interrupted):
                # Double-check for interruption (TTS also checks internally now)
                if self.interrupted.is_set():
                    logger.info(f"TTS interrupted after {chunks_played} chunks")
                    break

                # Send to voice chat output buffer
                await self.voice_chat.play_audio(audio_chunk)
                chunks_played += 1

            logger.info(f"TTS complete: {chunks_played} chunks played")

        except asyncio.CancelledError:
            logger.info(f"TTS cancelled after {chunks_played if 'chunks_played' in dir() else '?'} chunks")
            raise  # Re-raise to propagate cancellation
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
