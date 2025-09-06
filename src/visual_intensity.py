from __future__ import annotations
import pandas as pd
import torch
import open_clip
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from .config import RAW_DIR, PROC_DIR, CSV_PATH, OPENCLIP_MODEL, OPENCLIP_PRETRAINED, VISUAL_LABELS, FRAME_FPS
from .utils.video_utils import frame_iter

OUT_PATH = PROC_DIR / "visual_intensity.csv"

def main():
    df = pd.read_csv(CSV_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        OPENCLIP_MODEL, pretrained=OPENCLIP_PRETRAINED, device=device
    )
    tokenizer = open_clip.get_tokenizer(OPENCLIP_MODEL)
    text = tokenizer([f"a photo of {lab}" for lab in VISUAL_LABELS]).to(device)
    text_features = model.encode_text(text)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        vid = r["video_id"]
        mp4 = RAW_DIR / vid / f"{vid}.mp4"
        if not mp4.exists():
            continue
        try:
            sims = []
            for img in frame_iter(mp4, fps=FRAME_FPS):
                img_t = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    im_feat = model.encode_image(img_t)
                    im_feat /= im_feat.norm(dim=-1, keepdim=True)
                    logits = (im_feat @ text_features.T).softmax(dim=-1).squeeze(0)
                    sims.append(logits.detach().cpu().numpy())
            if len(sims) == 0:
                continue
            import numpy as np
            arr = np.vstack(sims)
            mean_scores = arr.mean(axis=0)
            result = {"video_id": vid, "VII": float(mean_scores.max())}
            for i, lab in enumerate(VISUAL_LABELS):
                result[f"p_{lab}"] = float(mean_scores[i])
            rows.append(result)
        except Exception as e:
            print("visual error", vid, e)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print("Wrote", OUT_PATH)

if __name__ == "__main__":
    main()
