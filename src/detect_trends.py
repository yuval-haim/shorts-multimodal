from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import ruptures as rpt
from .config import PROC_DIR, CSV_PATH

OUT_CPTS = PROC_DIR / "change_points.csv"

def detect_series_cpts(ts: pd.Series, pen: float = 3.0):
    x = ts.fillna(method="ffill").fillna(0).values.astype(float)
    if len(x) < 10:
        return []
    algo = rpt.Pelt(model="rbf").fit(x)
    # penalty controls # of breakpoints
    bkps = algo.predict(pen=pen)
    # format as indices
    return bkps

def main():
    feats = pd.read_parquet(PROC_DIR / "features.parquet")
    feats["publishedAt"] = pd.to_datetime(feats["publishedAt"], utc=True)
    feats["week"] = feats["publishedAt"].dt.to_period("W-MON").dt.start_time.dt.tz_localize("UTC")
    # Per-channel weekly HHI/SMI/VII
    weekly = feats.groupby(["channelId", "week"]).agg(
        HHI=("HHI", "mean"),
        VII=("VII", "mean"),
        toxicity=("toxicity", "mean"),
        views=("viewCount", "sum")
    ).reset_index()

    rows = []
    for ch, dfc in tqdm(weekly.groupby("channelId")):
        dfc = dfc.sort_values("week")
        for col in ["HHI", "VII", "toxicity"]:
            bkps = detect_series_cpts(dfc[col])
            for b in bkps:
                if b >= len(dfc): 
                    continue
                rows.append({
                    "channelId": ch,
                    "metric": col,
                    "week_cp": dfc.iloc[b]["week"]
                })
    out = pd.DataFrame(rows)
    out.to_csv(OUT_CPTS, index=False)
    weekly.to_csv(PROC_DIR / "weekly_metrics.csv", index=False)
    print("Wrote", OUT_CPTS, "and weekly_metrics.csv")

if __name__ == "__main__":
    main()
