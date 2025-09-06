# collect_youtube_shorts.py
import os, time, requests, csv, math, re
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = "AIzaSyBJ_FY1KJlxyKwJJWOKwzjrNBdVuP8ItmY"
assert API_KEY, "Put YTB_API_KEY in .env"

SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

# --- settings ---
QUERIES = [
    # EN
    "October 7", "Israel Hamas war", "Gaza war protest", "hostages rally", "ceasefire protest",
    # HE
    "7 באוקטובר", "מלחמת חרבות ברזל", "חטופים", "מחאת חטופים", "הפגנה עזה",
    # AR
    "٧ أكتوبر", "حرب غزة", "مظاهرة غزة", "إسرائيل حماس", "وقف إطلاق النار"
]
PUBLISHED_AFTER  = "2023-10-01T00:00:00Z"
PUBLISHED_BEFORE = "2025-08-31T23:59:59Z"
TARGET_COUNT = 1500   # collect extra, you can downsample to 150–200
MAX_PER_QUERY = 200  # cap per query to control balance
REGION_CODE = None   # e.g., "IL" or "US"; keep None for broader recall

def iso_duration_to_seconds(iso):
    m = re.match(r'PT(?:(\d+)M)?(?:(\d+)S)?', iso)
    if not m: 
        return 10**9
    mins = int(m.group(1) or 0)
    secs = int(m.group(2) or 0)
    return mins*60 + secs

def search_ids_for_query(q):
    ids = set()
    page = None
    while True:
        params = dict(
            key=API_KEY, part="snippet", q=q, type="video",
            publishedAfter=PUBLISHED_AFTER, publishedBefore=PUBLISHED_BEFORE,
            maxResults=50, videoDuration="short"
        )
        if REGION_CODE: params["regionCode"] = REGION_CODE
        if page: params["pageToken"] = page
        r = requests.get(SEARCH_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for it in data.get("items", []):
            ids.add(it["id"]["videoId"])
            if len(ids) >= MAX_PER_QUERY: break
        page = data.get("nextPageToken")
        if not page or len(ids) >= MAX_PER_QUERY:
            break
        time.sleep(0.1)
    return list(ids)

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def fetch_metadata(video_ids):
    out = []
    for batch in chunked(video_ids, 50):
        params = dict(
            key=API_KEY, part="snippet,contentDetails,statistics",
            id=",".join(batch)
        )
        r = requests.get(VIDEOS_URL, params=params, timeout=30)
        r.raise_for_status()
        for v in r.json().get("items", []):
            dur = iso_duration_to_seconds(v["contentDetails"]["duration"])
            # keep ≤ 90 seconds to approximate Shorts/Reels-style
            if dur <= 90:
                s = v["snippet"]; st = v.get("statistics", {})
                out.append({
                    "video_id": v["id"],
                    "publishedAt": s["publishedAt"],
                    "channelId": s["channelId"],
                    "channelTitle": s.get("channelTitle", ""),
                    "title": s.get("title","").replace("\n"," "),
                    "description": s.get("description","").replace("\n"," "),
                    "duration_sec": dur,
                    "viewCount": st.get("viewCount",""),
                    "likeCount": st.get("likeCount",""),
                    "commentCount": st.get("commentCount",""),
                    "tags": "|".join(s.get("tags", [])) if s.get("tags") else ""
                })
        time.sleep(0.2)
    return out

def main():
    all_ids = set()
    for q in QUERIES:
        # add try except
        try:
            ids = search_ids_for_query(q)
        except Exception as e:
            print(f"Error searching IDs for query '{q}': {e}")
            continue
        all_ids.update(ids)
        if len(all_ids) >= TARGET_COUNT * 2:  # oversample; we’ll filter later
            break

    meta = fetch_metadata(list(all_ids))
    # Sort by date, then sample top N evenly
    meta.sort(key=lambda x: x["publishedAt"])
    take = min(len(meta), TARGET_COUNT)
    selected = meta[:take]

    os.makedirs("data", exist_ok=True)
    with open("data/videos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(selected[0].keys()))
        w.writeheader()
        for row in selected:
            w.writerow(row)
    print(f"Wrote data/videos.csv with {len(selected)} rows")

if __name__ == "__main__":
    main()
