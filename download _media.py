# download_media.py
import os, csv, subprocess, sys
from yt_dlp import YoutubeDL
from tqdm import tqdm

CSV_PATH = "data/videos.csv"
OUT_DIR = "data/raw"

ydl_opts = {
    "outtmpl": os.path.join(OUT_DIR, "%(id)s", "%(id)s.%(ext)s"),
    "format": "bv*[height<=1080][fps<=60]+ba/b",
    "merge_output_format": "mp4",
    "writesubtitles": False,
    "quiet": False,
    "postprocessors": [
        {"key": "FFmpegExtractAudio", "preferredcodec": "wav", "preferredquality": "0"},
    ],
    "postprocessor_args": ["-ar", "16000", "-ac", "1"],  # 16k mono
    "cookies": "cookies.txt",
    "keepvideo": True
}
API_KEY = ""
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        # add tqdm bar
        for row in tqdm(csv.DictReader(f), desc="Downloading videos", unit="video"):
            vid = row["video_id"]
            url = f"https://www.youtube.com/watch?v={vid}"
            out_subdir = os.path.join(OUT_DIR, vid)
            os.makedirs(out_subdir, exist_ok=True)
            # add try except
            try:
                with YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
            except Exception as e:
                print(f"Error downloading {url}: {e}")

if __name__ == "__main__":
    main()
