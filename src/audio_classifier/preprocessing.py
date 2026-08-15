from pathlib import Path

import librosa
import numpy as np
import torch


TARGET_SAMPLE_RATE = 16000


def load_audio(audio_path: str, sample_rate: int = TARGET_SAMPLE_RATE):
    """
    Load an audio file and convert it to mono at the target sample rate.
    """

    path = Path(audio_path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, _ = librosa.load(
        path,
        sr=sample_rate,
        mono=True,
    )

    waveform = waveform.astype(np.float32)

    return waveform


def normalize_audio(waveform: np.ndarray) -> np.ndarray:
    """
    Normalize waveform amplitude.
    """

    max_value = np.max(np.abs(waveform))

    if max_value > 0:
        waveform = waveform / max_value

    return waveform.astype(np.float32)


def prepare_audio(audio_path: str):
    """
    Load, normalize, and convert audio to a PyTorch tensor.
    """

    waveform = load_audio(audio_path)

    waveform = normalize_audio(waveform)

    tensor = torch.tensor(
        waveform,
        dtype=torch.float32,
    )

    return tensor.unsqueeze(0)
