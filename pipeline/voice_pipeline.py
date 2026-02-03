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
        
        Uses sentence-level streaming: as each sentence completes from LLM,
        TTS generation starts immediately. Audio plays while later sentences
        are still being generated.

        Args:
            user_text: User's transcribed text
        """
        try:
            logger.info(f"LLM request (streaming): '{user_text}'")
            
            # State for sentence accumulation
            sentence_buffer = ""
            sentence_endings = {'.', '!', '?'}
            # Also break on newlines and long pauses (indicated by multiple spaces or dashes)
            
            # TTS tasks queue - we generate TTS concurrently with LLM streaming
            tts_queue = asyncio.Queue()
            tts_done = asyncio.Event()
            
            # Start TTS consumer task
            tts_consumer = asyncio.create_task(
                self._tts_consumer(tts_queue, tts_done)
            )
            
            self.is_speaking = True
            logger.info("🔊 STREAMING STARTED (is_speaking=True)")
            
            sentence_count = 0
            full_response = ""
            last_chunk_len = 0  # Track cumulative length to extract only new text
            
            # Stream tokens and accumulate sentences
            # NOTE: Clawdbot sends CUMULATIVE deltas (full response so far), not incremental
            async for chunk in self.claude.stream_response(user_text):
                if self.interrupted.is_set():
                    logger.info("INTERRUPTED: During LLM streaming - aborting")
                    break
                
                # Extract only the NEW portion of the cumulative response
                new_text = chunk[last_chunk_len:]
                last_chunk_len = len(chunk)
                
                if not new_text:
                    continue  # No new content in this delta
                
                full_response = chunk  # Store full cumulative response
                sentence_buffer += new_text  # Only add NEW text to buffer
                
                # Check for sentence boundaries
                # Look for sentence-ending punctuation followed by space or end
                while True:
                    # Find earliest sentence ending
                    earliest_end = -1
                    for punct in sentence_endings:
                        idx = sentence_buffer.find(punct)
                        if idx != -1:
                            # Check if followed by space, newline, or end of buffer
                            if idx + 1 >= len(sentence_buffer) or sentence_buffer[idx + 1] in ' \n\t':
                                if earliest_end == -1 or idx < earliest_end:
                                    earliest_end = idx
                    
                    # Also check for newlines as sentence breaks
                    newline_idx = sentence_buffer.find('\n')
                    if newline_idx != -1 and (earliest_end == -1 or newline_idx < earliest_end):
                        earliest_end = newline_idx
                    
                    if earliest_end == -1:
                        break  # No complete sentence yet
                    
                    # Extract the sentence (include the punctuation)
                    sentence = sentence_buffer[:earliest_end + 1].strip()
                    sentence_buffer = sentence_buffer[earliest_end + 1:].lstrip()
                    
                    if sentence:
                        sentence_count += 1
                        # Strip markdown and queue for TTS
                        clean_sentence = strip_markdown(sentence)
                        if clean_sentence:
                            logger.info(f"📝 Sentence {sentence_count}: '{clean_sentence[:50]}...'")
                            await tts_queue.put(clean_sentence)
            
            # Handle remaining text as final sentence
            remaining = sentence_buffer.strip()
            if remaining and not self.interrupted.is_set():
                clean_remaining = strip_markdown(remaining)
                if clean_remaining:
                    sentence_count += 1
                    logger.info(f"📝 Final sentence {sentence_count}: '{clean_remaining[:50]}...'")
                    await tts_queue.put(clean_remaining)
            
            # Signal TTS consumer that we're done adding sentences
            tts_done.set()
            
            # Wait for TTS consumer to finish
            if not self.interrupted.is_set():
                await tts_consumer
            else:
                tts_consumer.cancel()
                try:
                    await tts_consumer
                except asyncio.CancelledError:
                    pass
            
            logger.info(f"🔊 STREAMING COMPLETE: {sentence_count} sentences, {len(full_response)} chars total")
            logger.info(f"Full response: {full_response[:200]}...")
            
            # Wait for buffer to drain
            if not self.interrupted.is_set():
                drain_waited = 0
                while len(self.voice_chat.output_buffer) > 0:
                    if self.interrupted.is_set():
                        logger.info(f"Buffer drain interrupted after {drain_waited} waits")
                        break
                    await asyncio.sleep(0.05)
                    drain_waited += 1
                if drain_waited > 0:
                    logger.info(f"Buffer drained after {drain_waited * 50}ms")
            
            self.is_speaking = False
            logger.info("🔊 TTS complete (is_speaking=False)")

        except asyncio.CancelledError:
            logger.info("CANCELLED: LLM/TTS task was cancelled")
            self.is_speaking = False
            raise  # Re-raise to propagate
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            self.is_speaking = False

    async def _tts_consumer(self, sentence_queue: asyncio.Queue, done_event: asyncio.Event):
        """
        Consume sentences from queue and generate/play TTS.
        
        Runs concurrently with LLM streaming - generates TTS for each sentence
        as it becomes available.
        
        Args:
            sentence_queue: Queue of sentences to speak
            done_event: Event signaling no more sentences will be added
        """
        try:
            while True:
                # Check if we're done and queue is empty
                if done_event.is_set() and sentence_queue.empty():
                    break
                
                # Check for interruption
                if self.interrupted.is_set():
                    logger.info("TTS consumer interrupted")
                    break
                
                # Try to get next sentence
                try:
                    sentence = await asyncio.wait_for(sentence_queue.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue
                
                # Generate and play TTS for this sentence
                if sentence and not self.interrupted.is_set():
                    await self._speak_text(sentence)
                
        except asyncio.CancelledError:
            logger.info("TTS consumer cancelled")
            raise
        except Exception as e:
            logger.error(f"TTS consumer error: {e}", exc_info=True)

    async def _speak_text(self, text: str):
        """
        Generate TTS for a sentence and queue audio for playback.
        
        NOTE: Does NOT wait for buffer drain - that happens at the end of
        full response generation to enable overlapping sentence TTS.

        Args:
            text: Text to speak (typically a single sentence)
        """
        try:
            logger.info(f"TTS generating: '{text[:50]}...' ({len(text)} chars)")
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

            logger.info(f"TTS queued: {chunks_played} chunks for '{text[:30]}...'")

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

        # Clear buffers
        self.input_audio_buffer.clear()
        self.stt_buffer.clear()
        self.vad_accumulator.clear()

        logger.info("Voice pipeline stopped")
