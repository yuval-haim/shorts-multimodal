from __future__ import annotations
from pathlib import Path
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from torchvggish import vggish, vggish_input, vggish_params
from .config import RAW_DIR, PROC_DIR, CSV_PATH

OUT_PATH = PROC_DIR / "vggish_embeddings.npz"

def main():
    df = pd.read_csv(CSV_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = vggish()
    model.eval().to(device)

    embs = {}
    for _, r in tqdm(df.iterrows(), total=len(df)):
        vid = r["video_id"]
        wav = RAW_DIR / vid / f"{vid}.wav"
        if not wav.exists():
            continue
        try:
            ex = vggish_input.wavfile_to_examples(str(wav))
            ex = torch.from_numpy(ex).float().to(device)
            with torch.no_grad():
                feat = model(ex).cpu().numpy()
            embs[vid] = feat.mean(axis=0)  # 128-d mean pooling
        except Exception as e:
            print("vggish error", vid, e)
    # Save all to one NPZ for simplicity
    np.savez_compressed(OUT_PATH, **embs)
    print("Wrote", OUT_PATH)

if __name__ == "__main__":
    main()
