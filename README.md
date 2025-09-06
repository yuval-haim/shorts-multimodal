# Oct‑7 Shorts — Multimodal Analysis Pipeline

This repository extracts and analyzes **audio**, **vision**, and **text** signals from short‑form videos (YouTube Shorts/TikTok/IG Reels) related to the **Oct‑7 war period**. It produces monthly/weekly panels, interpretable topics, “visual buckets,” and trend/change‑point plots. The pipeline is reproducible end‑to‑end: **data gathering → preprocessing/cleaning → feature extraction → modeling → reporting**.

---

## Contents
- [Environment](#environment)
- [Project layout](#project-layout)
- [Data gathering](#data-gathering)
- [Preprocessing & filtering](#preprocessing--filtering)
- [Speech‑to‑text (ASR)](#speechto‑text-asr)
- [Feature extraction](#feature-extraction)
  - [Audio → HHI](#audio--hhi-hard-hitting-index)
  - [Vision → OpenCLIP + VII](#vision--openclip--vii-visual-intensity-index)
  - [Text → SBERT + TF‑IDF](#text--sbert--tf-idf)
- [Panels & aggregations](#panels--aggregations)
- [Models (the three we use)](#models-the-three-we-use)
  - [Model A — NLP Topic & Era‑Shift](#model-a--nlp-topic--era-shift)
  - [Model B — Vision Visual‑Intensity Shift](#model-b--vision-visual-intensity-shift)
  - [Model C — Audio Change‑Points & Era](#model-c--audio-change-points--era)
- [Run order (end‑to‑end)](#run-order-end-to-end)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Data schemas](#data-schemas)


---

## Environment

```bash
# Recommended: Python 3.10 / 3.11
conda create -n inflproj python=3.10 -y
conda activate inflproj

# Core
pip install numpy pandas pyarrow tqdm pyyaml python-dotenv requests
pip install matplotlib seaborn scikit-learn sentence-transformers

# ASR
pip install faster-whisper==1.0.3 soundfile

# Audio features
pip install librosa==0.10.1 torchvggish

# Vision
pip install open-clip-torch pillow opencv-python

# Change-point detection
pip install ruptures

# PyTorch (CPU shown; install the CUDA wheel from pytorch.org if you have a GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
> **Note:** `ruptures` currently has the best wheels for Python ≤3.11. Prefer a 3.10/3.11 env.

---

## Project layout

```
project/
  data/
    videos.csv                # collected metadata from YouTube API
    raw/                      # <video_id>/ folders with .mp4 + 16kHz mono .wav
    interim/                  # transcripts, sampled frames, temp artifacts
    processed/                # per-video features & monthly/weekly panels
  src/
    get_data_json.py          # YouTube API collection to JSON/CSV
    download_media.py         # yt-dlp download + WAV extraction
    asr_transcribe.py         # faster-whisper ASR
    audio_features.py         # librosa/VGGish + HHI
    visual_openclip.py        # frame sampling + OpenCLIP zero-shot
    text_filter.py            # keyword + semantic filter
    text_features.py          # SBERT embeddings + TF-IDF features
    build_panels.py           # monthly/weekly aggregations
    models_nlp.py             # Model A: topics & era classifier
    models_vision.py          # Model B: visual shift classifier
    models_audio.py           # Model C: HHI change-points & era
    detect_trends.py          # ruptures/PELT change-point utilities
  README.md
```

---

## Data gathering

1) **Collect candidate videos (YouTube Data API)**  
   - Query with EN/HE/AR terms connected to Oct‑7, Gaza, hostages, protests, ceasefire, etc.  
   - Restrict duration (e.g., ≤90s) to approximate “Shorts.”  
   - Output: `data/videos.csv` with `video_id`, `title`, `description`, published time, stats.

```bash
python -m src.get_data_json    # writes data/videos.csv
```

2) **Download media & extract WAV**  
   - Uses **yt‑dlp** to fetch MP4.  
   - Converts audio to **16kHz mono WAV** for ASR.

```bash
python -m src.download_media   # fills data/raw/<video_id>/<video_id>.mp4 + .wav
```

---

## Preprocessing & filtering

We keep a video if it passes **lexical** OR **semantic** relevance:

- **Keywords (EN/HE/AR):** e.g., “october 7 / 7 באוקטובר / ٧ أكتوبر”, “hostages / חטופים / الأسرى”, “ceasefire / הפסקת אש / هدنة”, “protest”, “idf/army”, etc.
- **Semantic similarity:** max cosine similarity between SBERT embedding of `(title + description + transcript)` and a small set of queries, e.g.,  
  “Videos about the October 7 events and the war in Israel and Gaza”,  
  “Protests, rallies, hostages, ceasefire, military escalation”.

```python
# keep if kw_score >= 2  OR  semantic_score >= 0.35  (tunable)
KEEP = (df["kw_score"] >= 2) | (df["sem_score"] >= 0.35)
filtered = df[KEEP]
filtered.to_parquet("data/processed/filtered.parquet", index=False)
```

---

## Speech‑to‑text (ASR)

- **faster‑whisper** (CTranslate2) transcribes WAV to JSON with per‑segment timestamps and language id.  
- Output: `data/interim/transcripts/<video_id>.json` and a merged table later.

```bash
python -m src.asr_transcribe
```

---

## Feature extraction

### Audio → HHI (Hard‑Hitting Index)

From **librosa** + optional **VGGish**, we compute:

- `rms_mean`, `onset_strength`, `spectral_flux`, `tempo_est`,  
- `percussive_energy`, `harmonic_energy`, `perc_harm_ratio`, …

**Definition (example):**  
`HHI = 0.35*z(rms_mean) + 0.30*z(onset_strength) + 0.20*z(perc_harm_ratio) + 0.15*z(spectral_flux)`  
All inputs are z‑scored per‑video (robust). Save to `data/processed/audio_features.parquet`.

```bash
python -m src.audio_features
```

### Vision → OpenCLIP + VII (Visual Intensity Index)

- Sample **N** frames per video (e.g., 6–12 evenly spaced).  
- Run **OpenCLIP** zero‑shot with prompts like:  
  `protest, flag, explosion, military, rally, fire, police, crowd, violence, smoke`  
  → produce per‑frame probabilities `p_*`. Aggregate per‑video by mean or max.
- **VII:** weighted index emphasizing destructive/violent cues:  
  `VII = 0.4*avg(smoke, explosion, fire, violence) + 0.2*military + 0.2*max(0, 0.6 - crowd) + 0.2*rally/flag` (example).  
- Output: `data/processed/vision_features.parquet` with all `p_*` + `VII`.

```bash
python -m src.visual_openclip
```

### Text → SBERT + TF‑IDF

- Build clean text: `title + description + transcript`.  
- Compute **SBERT** embeddings (multilingual MiniLM).  
- Build **TF‑IDF** (1–3‑grams, multilingual stoplists) for topic words and classifiers.  
- Output: `data/processed/text_features.parquet`.

```bash
python -m src.text_features
```

---

## Panels & aggregations

- Convert per‑video features to **monthly and weekly** panels: mean HHI, mean VII, visual bucket means, topic shares, counts per channel, etc.
- Optional: change‑point series with **ruptures/PELT** for `HHI` and `VII`.

```bash
python -m src.build_panels
```

---

## Models (full code models.ipynb)

### Model A — NLP Topic & Era‑Shift

**Goal:** discover interpretable topics and how they shift **Early vs Late** (split around `2024‑01‑31`, configurable).

- **Clustering:** SBERT embeddings → `MiniBatchKMeans` (K chosen by silhouette).  
- **Labeling:** class‑based TF‑IDF (**c‑TF‑IDF**) over cluster documents → top keywords per topic.  
- **Time series:** monthly topic **share**; plots: streamgraph, heatmap, **bump chart** for topic ranks.  
- **Classifier:** Early/Late logistic regression over TF‑IDF with `class_weight="balanced"` and group split by channel (no leakage).  
- **Outputs:**
  - `processed/nlp_topics.parquet` (topic id/label per video)
  - `processed/nlp_topic_monthly.parquet` (share per month)
  - `reports/fig_topic_stream.png`, `fig_topic_heatmap.png`, `fig_topic_bump.png`
  - `reports/nlp_era_clf.json` (precision/recall/AUC + top Early/Late terms)

### Model B — Vision Visual‑Intensity Shift

**Goal:** quantify how visuals shifted Early→Late (crowds vs destruction vs military) and identify predictive features.

- **Buckets:**  
  `V_crowd = mean(p_crowd, p_protest, p_flag, p_police, p_rally)`  
  `V_destruction = mean(p_smoke, p_explosion, p_fire, p_violence)`  
  `V_military = p_military`  
  also `VII`.
- **Classifier:** Early/Late logistic with `StandardScaler` + `class_weight="balanced"`.  
- **Plots:** small‑multiples “ribbons” per bucket, **slopegraph** Early→Late, coefficients barplot.  
- **Outputs:**
  - `processed/vision_buckets_monthly.parquet`
  - `reports/fig_visual_small_multiples.png`, `fig_visual_slope.png`, `fig_visual_coef.png`
  - `reports/vision_era_clf.json` (metrics + top features)

### Model C — Audio Change‑Points & Era

**Goal:** detect structural changes in **HHI** and summarize Early→Late differences.

- **Series:** compute weekly & monthly **HHI** means.  
- **Change‑points:** `ruptures.Pelt(model="rbf")` → `predict(pen=λ)`; adjust `pen` for granularity.  
- **Classifier (optional):** Early/Late logistic using audio features (`rms`, `onset_strength`, `spectral_flux`, `perc_harm_ratio`, `tempo_est`, …).  
- **Plots:** monthly/weekly HHI timeline, change‑point markers, Early→Late delta bar.  
- **Outputs:**
  - `processed/audio_monthly.parquet`, `audio_weekly.parquet`
  - `reports/fig_hhi_timeline.png`, `fig_hhi_changepoints.png`
  - `reports/audio_era_clf.json`
  - 

---

## Run order (end‑to‑end)

```bash
# 1) Gather
python -m src.get_data_json
python -m src.download_media

# 2) Preprocess & filter
python -m src.asr_transcribe
python -m src.text_filter

# 3) Features
python -m src.audio_features
python -m src.visual_openclip
python -m src.text_features

# 4) Panels & models
python -m src.build_panels
python -m src.models_nlp
python -m src.models_vision
python -m src.models_audio

# 5) (optional) Trend detection utilities
python -m src.detect_trends
```

---

## Configuration

Edit `src/config.py`:
- Paths: `DATA_DIR`, `RAW_DIR`, `INTERIM_DIR`, `PROCESSED_DIR`, `REPORTS_DIR`
- Frame sampling: `N_FRAMES`, `FRAME_SIZE`
- OpenCLIP: backbone name, prompts
- Audio: feature set & HHI weights
- NLP: topic K search range, stopword lists, TF‑IDF params
- Era split date, random seeds

Secrets:
- `.env` for `YTB_API_KEY`, optional `PERSPECTIVE_API_KEY` (if using Perspective).

---

## Outputs

- **Processed data**
  - `processed/features_audio.parquet`
  - `processed/features_vision.parquet`
  - `processed/features_text.parquet`
  - `processed/monthly_panel.parquet`, `processed/weekly_panel.parquet`
- **Reports & figures**
  - Topic streamgraph / heatmap / bump chart
  - Visual bucket small‑multiples / slopegraph / coefficients
  - HHI timeline / change‑points
  - Early↔Late metrics JSON for each model

---

## Data schemas

**Per‑video (joined)**
```
video_id, channelId, publishedAt, duration, lang,
title, description, transcript,
HHI, rms_mean, onset_strength, spectral_flux, perc_harm_ratio, ...,
p_protest, p_flag, p_explosion, p_military, p_rally, p_fire, p_police, p_crowd, p_violence, p_smoke, VII,
sbert_emb_*(optional),
topic_id, topic_label(optional)
```

**Monthly/Weekly panels**
```
month|week, n_videos, HHI_mean, VII_mean,
V_crowd, V_destruction, V_military,
topic_share_T0..Tk, ...
```


