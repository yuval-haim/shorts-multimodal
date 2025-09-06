from __future__ import annotations
import json
from pathlib import Path
from tqdm import tqdm
from faster_whisper import WhisperModel
import pandas as pd
from .config import RAW_DIR, INTERIM_DIR, CSV_PATH, WHISPER_MODEL, WHISPER_BEAM

OUT_DIR = INTERIM_DIR / "transcripts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def transcribe_one(wav_path: Path, model: WhisperModel):
    segments_out = []
    # word_timestamps=True is supported by faster-whisper
    segments, info = model.transcribe(str(wav_path), beam_size=WHISPER_BEAM, word_timestamps=True)
    lang = info.language or ""
    for seg in segments:
        seg_dict = {
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text,
            "words": [
                {"start": float(w.start), "end": float(w.end), "word": w.word}
                for w in (seg.words or [])
            ]
        }
        segments_out.append(seg_dict)
    return {"language": lang, "segments": segments_out}

def main():
    df = pd.read_csv(CSV_PATH)
    model = WhisperModel(WHISPER_MODEL, device="auto", compute_type="auto")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        vid = row["video_id"]
        wav = RAW_DIR / vid / f"{vid}.wav"
        if not wav.exists():
            continue
        out_path = OUT_DIR / f"{vid}.json"
        if out_path.exists():
            continue
        try:
            result = transcribe_one(wav, model)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False)
        except Exception as e:
            print("ASR error", vid, e)

if __name__ == "__main__":
    main()
