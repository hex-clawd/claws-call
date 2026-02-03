"""Voice Activity Detection using Silero VAD."""

import logging
import torch
import numpy as np
from typing import Optional
import config

logger = logging.getLogger(__name__)


class VAD:
    """Voice Activity Detection using Silero VAD."""

    def __init__(self):
        """Initialize Silero VAD model."""
        try:
            logger.info("Loading Silero VAD model...")
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )

            # Extract utility functions
            (self.get_speech_timestamps,
             self.save_audio,
             self.read_audio,
             self.VADIterator,
             self.collect_chunks) = self.utils

            # Configuration
            self.sample_rate = config.SAMPLE_RATE_WHISPER  # 16kHz
            self.threshold = config.VAD_SPEECH_THRESHOLD
            self.silence_threshold_ms = config.VAD_SILENCE_THRESHOLD_MS
            self.min_speech_duration_ms = config.VAD_MIN_SPEECH_DURATION_MS

            # State tracking
            self.is_speech = False
            self.silence_frames = 0
            self.speech_frames = 0
            self.frame_duration_ms = 50  # Using 50ms frames for safety margin (Silero needs min 32ms)

            logger.info(f"Silero VAD initialized successfully")
            logger.info(f"Config: threshold={self.threshold}, silence={self.silence_threshold_ms}ms, min_speech={self.min_speech_duration_ms}ms")

        except Exception as e:
            logger.error(f"Failed to initialize Silero VAD: {e}")
            raise

    def check_speech(self, audio_chunk: np.ndarray) -> float:
        """
        Check if audio chunk contains speech.

        Args:
            audio_chunk: Audio data as numpy array (float32, 16kHz)

        Returns:
            Speech probability (0.0 to 1.0)
        """
        try:
            # Convert to torch tensor if needed
            if isinstance(audio_chunk, np.ndarray):
                audio_tensor = torch.from_numpy(audio_chunk)
            else:
                audio_tensor = audio_chunk

            # Ensure float32
            if audio_tensor.dtype != torch.float32:
                audio_tensor = audio_tensor.float()

            # Run VAD
            with torch.no_grad():
                speech_prob = self.model(audio_tensor, self.sample_rate).item()

            return speech_prob

        except Exception as e:
            logger.error(f"VAD check failed: {e}")
            return 0.0

    def is_speech_active(self, audio_chunk: np.ndarray) -> bool:
        """
        Check if speech is currently active in the audio chunk.

        Args:
            audio_chunk: Audio data as numpy array (float32, 16kHz)

        Returns:
            True if speech is detected, False otherwise
        """
        speech_prob = self.check_speech(audio_chunk)
        return speech_prob > self.threshold

    def update_state(self, audio_chunk: np.ndarray) -> tuple[bool, bool]:
        """
        Update VAD state and detect turn boundaries.

        Args:
            audio_chunk: Audio data as numpy array (float32, 16kHz)

        Returns:
            Tuple of (is_speech_now, turn_ended)
            - is_speech_now: True if speech is detected in this chunk
            - turn_ended: True if silence threshold exceeded (end of turn)
        """
        speech_prob = self.check_speech(audio_chunk)
        is_speech_now = speech_prob > self.threshold

        turn_ended = False

        if is_speech_now:
            self.speech_frames += 1
            self.silence_frames = 0
            if not self.is_speech:
                # Speech started
                logger.debug("Speech started")
                self.is_speech = True
        else:
            self.silence_frames += 1
            if self.is_speech:
                # Check if silence exceeds threshold
                silence_ms = self.silence_frames * self.frame_duration_ms
                if silence_ms >= self.silence_threshold_ms:
                    # Check if we had enough speech
                    speech_ms = self.speech_frames * self.frame_duration_ms
                    if speech_ms >= self.min_speech_duration_ms:
                        logger.debug(f"Turn ended: {speech_ms}ms speech, {silence_ms}ms silence")
                        turn_ended = True
                    # Reset state
                    self.is_speech = False
                    self.speech_frames = 0
                    self.silence_frames = 0

        return is_speech_now, turn_ended

    def reset(self):
        """Reset VAD state."""
        self.is_speech = False
        self.silence_frames = 0
        self.speech_frames = 0
        logger.debug("VAD state reset")

    def get_speech_timestamps_from_audio(self, audio: np.ndarray) -> list:
        """
        Get speech timestamps from full audio (non-streaming mode).

        Args:
            audio: Full audio as numpy array (float32, 16kHz)

        Returns:
            List of speech segments with start/end timestamps
        """
        try:
            if isinstance(audio, np.ndarray):
                audio_tensor = torch.from_numpy(audio)
            else:
                audio_tensor = audio

            if audio_tensor.dtype != torch.float32:
                audio_tensor = audio_tensor.float()

            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                sampling_rate=self.sample_rate,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.silence_threshold_ms
            )

            return speech_timestamps

        except Exception as e:
            logger.error(f"Failed to get speech timestamps: {e}")
            return []
