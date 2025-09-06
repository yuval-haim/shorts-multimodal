from __future__ import annotations
import numpy as np
import soundfile as sf
from pathlib import Path

def load_mono_wav(path: Path, target_sr: int = 16000):
    import librosa
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    return y, sr

def robust_rms(y: np.ndarray, frame_length: int = 2048, hop_length: int = 512):
    import librosa
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    return rms

def dynamic_range_db(rms: np.ndarray) -> float:
    # 95th - 5th percentile in dB as a measure of punch/compression
    rms_db = 20 * np.log10(np.maximum(rms, 1e-9))
    lo = np.percentile(rms_db, 5)
    hi = np.percentile(rms_db, 95)
    return float(hi - lo)
