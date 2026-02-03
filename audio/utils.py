"""Audio utilities for format conversion and resampling."""

import logging
import numpy as np
from scipy import signal
import struct
import config

logger = logging.getLogger(__name__)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio from one sample rate to another.

    Args:
        audio: Audio data as numpy array
        orig_sr: Original sample rate
        target_sr: Target sample rate

    Returns:
        Resampled audio as numpy array
    """
    if orig_sr == target_sr:
        return audio

    try:
        # Calculate number of output samples
        num_samples = int(len(audio) * target_sr / orig_sr)

        # Use scipy's resample for high-quality resampling
        resampled = signal.resample(audio, num_samples)

        logger.debug(f"Resampled audio from {orig_sr}Hz to {target_sr}Hz: {len(audio)} -> {len(resampled)} samples")
        return resampled

    except Exception as e:
        logger.error(f"Resampling failed: {e}")
        raise


def pcm_bytes_to_float32(pcm_bytes: bytes, sample_width: int = 2) -> np.ndarray:
    """
    Convert PCM bytes to float32 numpy array.

    Args:
        pcm_bytes: Raw PCM audio bytes
        sample_width: Bytes per sample (2 for 16-bit, 4 for 32-bit)

    Returns:
        Audio as float32 numpy array normalized to [-1.0, 1.0]
    """
    try:
        # Convert bytes to int16 or int32
        if sample_width == 2:
            # 16-bit PCM
            audio_int = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_float = audio_int.astype(np.float32) / 32768.0
        elif sample_width == 4:
            # 32-bit PCM
            audio_int = np.frombuffer(pcm_bytes, dtype=np.int32)
            audio_float = audio_int.astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        return audio_float

    except Exception as e:
        logger.error(f"PCM to float32 conversion failed: {e}")
        raise


def float32_to_pcm_bytes(audio: np.ndarray, sample_width: int = 2) -> bytes:
    """
    Convert float32 numpy array to PCM bytes.

    Args:
        audio: Audio as float32 numpy array (should be in range [-1.0, 1.0])
        sample_width: Bytes per sample (2 for 16-bit, 4 for 32-bit)

    Returns:
        Raw PCM audio bytes
    """
    try:
        # Clip to valid range
        audio = np.clip(audio, -1.0, 1.0)

        if sample_width == 2:
            # Convert to 16-bit PCM
            audio_int = (audio * 32767.0).astype(np.int16)
        elif sample_width == 4:
            # Convert to 32-bit PCM
            audio_int = (audio * 2147483647.0).astype(np.int32)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        return audio_int.tobytes()

    except Exception as e:
        logger.error(f"Float32 to PCM conversion failed: {e}")
        raise


def convert_telegram_to_whisper(pcm_48k_bytes: bytes, stereo: bool = True) -> np.ndarray:
    """
    Convert Telegram audio (48kHz PCM) to Whisper format (16kHz float32).

    Args:
        pcm_48k_bytes: Raw PCM bytes from Telegram (16-bit, 48kHz)
        stereo: If True, input is stereo and will be converted to mono

    Returns:
        Audio as float32 numpy array (16kHz, normalized, mono)
    """
    try:
        # Convert bytes to float32
        audio_float = pcm_bytes_to_float32(pcm_48k_bytes, sample_width=2)

        # Convert stereo to mono if needed (pytgcalls sends stereo by default)
        if stereo and len(audio_float) >= 2:
            # Interleaved stereo: [L0, R0, L1, R1, ...] -> average channels
            left = audio_float[0::2]
            right = audio_float[1::2]
            audio_float = (left + right) / 2.0
            logger.debug(f"Converted stereo to mono: {len(left)} samples per channel")

        # Resample 48kHz -> 16kHz
        audio_16k = resample_audio(
            audio_float,
            orig_sr=config.SAMPLE_RATE_TG,
            target_sr=config.SAMPLE_RATE_WHISPER
        )

        return audio_16k

    except Exception as e:
        logger.error(f"Telegram to Whisper conversion failed: {e}")
        raise


def convert_whisper_to_telegram(audio_16k: np.ndarray) -> bytes:
    """
    Convert Whisper format audio (16kHz float32) to Telegram format (48kHz PCM).

    Args:
        audio_16k: Audio as float32 numpy array (16kHz)

    Returns:
        Raw PCM bytes for Telegram (16-bit, 48kHz, mono)
    """
    try:
        # Resample 16kHz -> 48kHz
        audio_48k = resample_audio(
            audio_16k,
            orig_sr=config.SAMPLE_RATE_WHISPER,
            target_sr=config.SAMPLE_RATE_TG
        )

        # Convert to PCM bytes
        pcm_bytes = float32_to_pcm_bytes(audio_48k, sample_width=2)

        return pcm_bytes

    except Exception as e:
        logger.error(f"Whisper to Telegram conversion failed: {e}")
        raise


def generate_silence(duration_ms: int, sample_rate: int = 48000, sample_width: int = 2) -> bytes:
    """
    Generate silence as PCM bytes.

    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate in Hz
        sample_width: Bytes per sample

    Returns:
        Silence as PCM bytes
    """
    num_samples = int(duration_ms * sample_rate / 1000)
    silence = np.zeros(num_samples, dtype=np.float32)
    return float32_to_pcm_bytes(silence, sample_width)


def calculate_audio_duration(pcm_bytes: bytes, sample_rate: int, sample_width: int = 2) -> float:
    """
    Calculate duration of PCM audio in seconds.

    Args:
        pcm_bytes: Raw PCM bytes
        sample_rate: Sample rate in Hz
        sample_width: Bytes per sample

    Returns:
        Duration in seconds
    """
    num_samples = len(pcm_bytes) // sample_width
    duration = num_samples / sample_rate
    return duration


def split_audio_chunks(audio: np.ndarray, chunk_duration_ms: int, sample_rate: int) -> list[np.ndarray]:
    """
    Split audio into fixed-duration chunks.

    Args:
        audio: Audio as numpy array
        chunk_duration_ms: Chunk duration in milliseconds
        sample_rate: Sample rate in Hz

    Returns:
        List of audio chunks as numpy arrays
    """
    samples_per_chunk = int(chunk_duration_ms * sample_rate / 1000)
    chunks = []

    for i in range(0, len(audio), samples_per_chunk):
        chunk = audio[i:i + samples_per_chunk]
        if len(chunk) == samples_per_chunk:
            chunks.append(chunk)
        else:
            # Pad last chunk if needed
            padded = np.zeros(samples_per_chunk, dtype=audio.dtype)
            padded[:len(chunk)] = chunk
            chunks.append(padded)

    return chunks
