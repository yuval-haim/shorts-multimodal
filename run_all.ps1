# Run the full pipeline (Windows PowerShell)
# Usage: in PowerShell from influencer_pipeline\
#   conda activate inflproj
#   ./run_all.ps1

python -m src.asr_transcribe
python -m src.audio_features
python -m src.visual_intensity
python -m src.text_cues
python -m src.embeddings_vggish
python -m src.merge_datasets
python -m src.build_panel
python -m src.detect_trends
