from __future__ import annotations
import os, json, time
from pathlib import Path
import pandas as pd
import requests
from tqdm import tqdm
from .config import INTERIM_DIR, PROC_DIR, CSV_PATH, PERSPECTIVE_API_KEY, NRC_LEXICON_PATH, EMFD_LEXICON_PATH

OUT_PATH = PROC_DIR / "text_cues.csv"

def load_transcript_text(vid: str) -> str:
    p = INTERIM_DIR / "transcripts" / f"{vid}.json"
    if not p.exists():
        return ""
    obj = json.loads(p.read_text(encoding="utf-8"))
    return " ".join(seg["text"] for seg in obj.get("segments", []))

def perspective_score(text: str, attr="TOXICITY") -> float | None:
    if not PERSPECTIVE_API_KEY:
        return None
    if not text.strip():
        return None
    url = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"
    data = {
        "comment": {"text": text[:2000]},  # API limit, keep short summary
        "languages": ["en"],
        "requestedAttributes": {attr: {}},
        "doNotStore": True,
    }
    try:
        r = requests.post(url, params={"key": PERSPECTIVE_API_KEY}, json=data, timeout=10)
        r.raise_for_status()
        js = r.json()
        return js["attributeScores"][attr]["summaryScore"]["value"]
    except Exception:
        return None

def main():
    df = pd.read_csv(CSV_PATH)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        vid = r["video_id"]
        txt = load_transcript_text(vid)
        tox = perspective_score(txt, "TOXICITY")
        ins = perspective_score(txt, "INSULT")
        thr = perspective_score(txt, "THREAT")
        rows.append({"video_id": vid, "toxicity": tox, "insult": ins, "threat": thr})
        time.sleep(0.2)  # be gentle to API
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print("Wrote", OUT_PATH)

if __name__ == "__main__":
    main()
