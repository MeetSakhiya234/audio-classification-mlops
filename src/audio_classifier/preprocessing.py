from pathlib import Path
import io

import librosa
import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 16000


def load_audio(
    audio_path: str,
    sample_rate: int = TARGET_SAMPLE_RATE,
):
    """
    Load an audio file and convert it to mono
    at the target sample rate.
    """

    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {path}"
        )

    waveform, _ = librosa.load(
        path,
        sr=sample_rate,
        mono=True,
    )

    return waveform.astype(np.float32)


def normalize_audio(
    waveform: np.ndarray,
) -> np.ndarray:
    """
    Normalize waveform amplitude.
    """

    max_value = np.max(
        np.abs(waveform)
    )

    if max_value > 0:
        waveform = waveform / max_value

    return waveform.astype(np.float32)


def prepare_audio(audio_path: str):
    """
    Load, normalize, and prepare audio
    as a NumPy array for ONNX Runtime.
    """

    waveform = load_audio(audio_path)

    waveform = normalize_audio(waveform)

    return np.expand_dims(
        waveform,
        axis=0,
    ).astype(np.float32)


def prepare_audio_bytes(
    audio_bytes: bytes,
):
    """
    Load audio directly from bytes, normalize it,
    and prepare it as a NumPy array for ONNX Runtime.
    """

    waveform, sample_rate = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32",
    )

    if waveform.ndim > 1:
        waveform = np.mean(
            waveform,
            axis=1,
        )

    if sample_rate != TARGET_SAMPLE_RATE:
        waveform = librosa.resample(
            waveform,
            orig_sr=sample_rate,
            target_sr=TARGET_SAMPLE_RATE,
        )

    waveform = normalize_audio(waveform)

    return np.expand_dims(
        waveform,
        axis=0,
    ).astype(np.float32)
