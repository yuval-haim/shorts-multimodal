import os, math, time, requests, pandas as pd
from dotenv import load_dotenv
from pathlib import Path

from .config import DATA_DIR, PROC_DIR, CSV_PATH

# Fetch current channel statistics (subscribers, videoCount, viewCount)
# and save to processed/channels.csv

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def main():
    load_dotenv()
    API_KEY = os.getenv("YTB_API_KEY")
    if not API_KEY:
        print("Set YTB_API_KEY in your environment or .env")
        return
    meta = pd.read_csv(CSV_PATH, dtype={"channelId": str})
    chs = sorted(meta["channelId"].dropna().unique().tolist())
    rows = []
    for batch in chunked(chs, 50):
        url = "https://www.googleapis.com/youtube/v3/channels"
        params = dict(key=API_KEY, part="snippet,statistics", id=",".join(batch))
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        for it in r.json().get("items", []):
            st = it.get("statistics", {})
            sn = it.get("snippet", {})
            rows.append({
                "channelId": it["id"],
                "channelTitle": sn.get("title", ""),
                "country": sn.get("country", ""),
                "subscribers": int(st.get("subscriberCount", 0)) if st.get("hiddenSubscriberCount") is not True else None,
                "videoCount": int(st.get("videoCount", 0)) if st.get("videoCount") is not None else None,
                "channelViews": int(st.get("viewCount", 0)) if st.get("viewCount") is not None else None,
            })
        time.sleep(0.2)
    out = pd.DataFrame(rows)
    outp = PROC_DIR / "channels.csv"
    out.to_csv(outp, index=False)
    print("Wrote", outp)

if __name__ == "__main__":
    main()
