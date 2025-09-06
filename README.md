# Oct-7 videos multimodal anlysis 
This is a out pipeline we used to extract the features for our multimodal anlysis from the youtube shorts we mined:
- **ASR** with faster-whisper → word timestamps + language
- **Audio features** (librosa) → Hard-Hitting Index (HHI)
- **Visual intensity** (OpenCLIP zero-shot on frames)
- **Text cues** (Perspective API + lexicons) → Strong Message Index (SMI)
- **Weekly creator panel** + **change-point detection** (ruptures)

## 0) Create environment (Windows-friendly)

```powershell
# Create and activate conda env (recommended)
conda create -n inflproj python=3.10 -y
conda activate inflproj

# Core deps
pip install numpy pandas pyarrow tqdm pyyaml python-dotenv
pip install faster-whisper==1.0.3  # CPU/GPU via CTranslate2
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu  # or CUDA build
pip install open-clip-torch pillow opencv-python
pip install librosa==0.10.1 soundfile
pip install ruptures
pip install torchvggish  # VGGish embeddings
# Optional (Perspective API)
pip install google-api-python-client requests
```

> If you have a **CUDA** GPU, install the CUDA build of PyTorch from pytorch.org and faster-whisper will use it automatically.

## 1) Project layout

```
pipeline/
  data/
    videos.csv            # videos metdata after download
    raw/                  #  video_id/ subfolders with .mp4 + .wav 
    interim/              # transcripts, frame samples, etc.
    processed/            # feature tables, weekly panel
  src/
    config.py
    build_panel.py
    asr_transcribe.py
    audio_features.py
    visual_intensity.py
    text_cues.py
    embeddings_vggish.py
    merge_datasets.py
    detect_trends.py
    utils/
      audio_utils.py
      video_utils.py
      time_utils.py
README.md
```

## 2) Order of execution

```powershell
# From pipeline/
# A) Transcribe audio
python -m src.asr_transcribe

# B) Audio features (+ HHI)
python -m src.audio_features

# C) Visual intensity (OpenCLIP zero-shot on frames)
python -m src.visual_intensity

# D) Text cues (Perspective API + lexicons if available)
#    Set PERSPECTIVE_API_KEY in your environment or .env
python -m src.text_cues

# E) Merge per-video features → processed/features.parquet
python -m src.merge_datasets

# F) Weekly creator panel + change-points
python -m src.build_panel
python -m src.detect_trends
```

## 3) Configuration

Edit `src/config.py` to point to your folders and choose labels for visual zero-shot, model sizes, etc.

