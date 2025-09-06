from __future__ import annotations
import pandas as pd
from pathlib import Path
from .config import PROC_DIR, CSV_PATH
from .utils.time_utils import to_week

OUT_PANEL = PROC_DIR / "weekly_panel.csv"

def main():
    meta = pd.read_csv(CSV_PATH, parse_dates=["publishedAt"])
    # Safe numeric
    for col in ["viewCount", "likeCount", "commentCount"]:
        if col in meta.columns:
            meta[col] = pd.to_numeric(meta[col], errors="coerce").fillna(0)
    meta["week"] = to_week(meta["publishedAt"])
    grp = meta.groupby(["channelId", "week"]).agg(
        videos=("video_id", "count"),
        views=("viewCount", "sum"),
        likes=("likeCount", "sum"),
        comments=("commentCount", "sum"),
        avg_dur=("duration_sec", "mean")
    ).reset_index()
    grp.to_csv(OUT_PANEL, index=False)
    print("Wrote", OUT_PANEL)

if __name__ == "__main__":
    main()
