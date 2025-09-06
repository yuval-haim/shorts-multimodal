from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from .config import PROC_DIR, CSV_PATH

def load_npz_embeddings(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["video_id"])
    npz = np.load(path, allow_pickle=True)
    rows = []
    for vid in npz.files:
        vec = npz[vid].tolist()
        rows.append({"video_id": vid, **{f"vgg_{i}": v for i, v in enumerate(vec)}})
    return pd.DataFrame(rows)

def main():
    meta = pd.read_csv(CSV_PATH)
    audio = pd.read_csv(PROC_DIR / "audio_features.csv", dtype={"video_id": str}) if (PROC_DIR / "audio_features.csv").exists() else pd.DataFrame()
    visual = pd.read_csv(PROC_DIR / "visual_intensity.csv", dtype={"video_id": str}) if (PROC_DIR / "visual_intensity.csv").exists() else pd.DataFrame()
    text = pd.read_csv(PROC_DIR / "text_cues.csv", dtype={"video_id": str}) if (PROC_DIR / "text_cues.csv").exists() else pd.DataFrame()
    vgg = load_npz_embeddings(PROC_DIR / "vggish_embeddings.npz")

    df = meta.merge(audio, on="video_id", how="left")              .merge(visual, on="video_id", how="left")              .merge(text, on="video_id", how="left")              .merge(vgg, on="video_id", how="left")

    outp = PROC_DIR / "features.parquet"
    df.to_parquet(outp, index=False)
    print("Wrote", outp)

if __name__ == "__main__":
    main()
