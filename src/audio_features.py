from __future__ import annotations
import numpy as np, json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import librosa
from .config import RAW_DIR, PROC_DIR, CSV_PATH, HHI_WEIGHTS
from .utils.audio_utils import load_mono_wav, robust_rms, dynamic_range_db

OUT_PATH = PROC_DIR / "audio_features.csv"

def features_for_wav(wav_path: Path):
    y, sr = load_mono_wav(wav_path, 16000)
    hop = 512
    # RMS loudness
    rms = robust_rms(y, hop_length=hop)
    rms_mean = float(np.mean(rms))
    # Spectral flux via onset strength
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    spectral_flux = float(np.mean(onset_env))
    # Onset rate (per second)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_rate = float(len(onsets) / (len(y)/sr + 1e-9))
    # Spectral centroid (proxy for brightness)
    sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(sc))
    # Tempo (global)
    tempo = float(librosa.feature.tempo(onset_envelope=onset_env, sr=sr)[0])
    tempo_norm = tempo / 220.0  # rough normalization vs fast EDM/HH tempos
    # Dynamic range (dB, lower = more compressed => more "hard-hitting")
    dr = dynamic_range_db(rms)

    feats = dict(
        rms=rms_mean,
        spectral_flux=spectral_flux,
        onset_rate=onset_rate,
        centroid=centroid_mean,
        tempo=tempo,
        tempo_norm=tempo_norm,
        dynamic_range_db=dr,
    )
    return feats

def compute_hhi(df_feats: pd.DataFrame) -> pd.Series:
    # z-score columns, then weighted sum
    z = (df_feats - df_feats.mean()) / (df_feats.std(ddof=0) + 1e-9)
    hhi = (
        HHI_WEIGHTS["rms"] * z["rms"] +
        HHI_WEIGHTS["spectral_flux"] * z["spectral_flux"] +
        HHI_WEIGHTS["onset_rate"] * z["onset_rate"] +
        HHI_WEIGHTS["tempo_norm"] * z["tempo_norm"] +
        HHI_WEIGHTS["dynamic_range"] * z["dynamic_range_db"]
    )
    return hhi

def main():
    meta = pd.read_csv(CSV_PATH)
    rows = []
    for _, r in tqdm(meta.iterrows(), total=len(meta)):
        vid = r["video_id"]
        wav = RAW_DIR / vid / f"{vid}.wav"
        if not wav.exists():
            continue
        try:
            f = features_for_wav(wav)
            f["video_id"] = vid
            rows.append(f)
        except Exception as e:
            print("audio feat error", vid, e)
    df = pd.DataFrame(rows).set_index("video_id")
    if len(df) == 0:
        print("No audio found.")
        return
    df["HHI"] = compute_hhi(df)
    df.reset_index().to_csv(OUT_PATH, index=False)
    print("Wrote", OUT_PATH)

if __name__ == "__main__":
    main()
