# BioGuard AI: Physical & Digital Biometric Spoofing Detection for Secure E-Transactions

## 1. Installation & Execution Commands
```bash
# 1. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Run automated unit & integration tests
pytest -v

# 4. Launch the Streamlit application
streamlit run app.py
```

## 2. Dataset Integration Procedure
Academic biometric datasets require signed End-User License Agreements (EULAs). After obtaining approved access:
1. **Physical PAD (CelebA-Spoof / OULU-NPU / Replay-Attack):**
   Place cropped face images under `datasets/data/pad/` and create `datasets/data/pad/manifest.csv`:
   ```csv
   filepath,label,subject_id,attack_subtype
   live/0001.jpg,0,subj_01,live
   print/0002.jpg,1,subj_01,print
   replay/0003.jpg,1,subj_02,replay
   ```
   Train & evaluate Model A:
   ```bash
   python -m training.train_pad --manifest datasets/data/pad/manifest.csv --epochs 15
   python -m evaluation.evaluate_models --task pad --manifest datasets/data/pad/manifest.csv
   ```

2. **Digital Deepfake (FaceForensics++ / Celeb-DF / DFDC):**
   Extract face frames under `datasets/data/deepfake/` and create `datasets/data/deepfake/manifest.csv`:
   ```csv
   filepath,label,subject_id,attack_subtype
   real/vid01_f01.jpg,0,video_01,real
   fake/vid02_f01.jpg,1,video_02,faceswap
   ```
   Train & evaluate Model B:
   ```bash
   python -m training.train_deepfake --manifest datasets/data/deepfake/manifest.csv --epochs 12
   python -m evaluation.evaluate_models --task deepfake --manifest datasets/data/deepfake/manifest.csv
   ```

## 3. Troubleshooting
- **No face detected:** Ensure frontal illumination and a minimum face resolution of `64x64` pixels.
- **CUDA Out of Memory:** Reduce `--batch-size` from `32` to `8` or `16` during training.
- **Weight Status Warning in UI:** Until `.pth` checkpoints are generated via the training scripts above, the app transparently runs in deterministic CV Forensic Signal Mode rather than making unverified neural predictions.
