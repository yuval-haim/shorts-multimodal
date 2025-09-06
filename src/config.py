import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present (for API keys like PERSPECTIVE_API_KEY)
load_dotenv()

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROC_DIR = DATA_DIR / "processed"
CSV_PATH = DATA_DIR / "videos.csv"

# --- Models & settings ---
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "medium")  # tiny/base/small/medium/large-v3
WHISPER_BEAM = int(os.getenv("WHISPER_BEAM", "5"))

# OpenCLIP
OPENCLIP_MODEL = os.getenv("OPENCLIP_MODEL", "ViT-B-32")
OPENCLIP_PRETRAINED = os.getenv("OPENCLIP_PRETRAINED", "laion2b_s34b_b79k")
# Label set for visual 'intensity'
VISUAL_LABELS = [
    "protest", "flag", "explosion", "military", "rally", "fire", "police", "crowd",
    "violence", "march", "demonstration", "smoke"
]

# Perspective API
PERSPECTIVE_API_KEY = os.getenv("PERSPECTIVE_API_KEY", None)

# NRC / eMFD lexicon paths (optional; put files if you have them)
NRC_LEXICON_PATH = os.getenv("NRC_LEXICON_PATH", "")
EMFD_LEXICON_PATH = os.getenv("EMFD_LEXICON_PATH", "")

# Frame sampling
FRAME_FPS = float(os.getenv("FRAME_FPS", "1"))  # frames per second to sample

# Hard-Hitting Index weights
HHI_WEIGHTS = {
    "rms": 1.0,
    "spectral_flux": 1.0,
    "onset_rate": 1.0,
    "tempo_norm": 1.0,
    "dynamic_range": -1.0
}
