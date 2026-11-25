# Viva-Voce Questions & Answers

**Q1: Why use two separate models (EfficientNet-B0 + LBP and Xception + FFT) instead of a single binary classifier?**
**A:** Physical presentation attacks and digital deepfakes leave fundamentally different forensic traces. Printed photos and screen replays exhibit macro-surface cues such as specular glare, color gamut compression, and paper/pixel micro-textures (captured by LBP + EfficientNet-B0). Digital deepfakes lack physical camera recapture artifacts and instead exhibit subtle facial boundary blending and high-frequency upsampling anomalies in the Fourier domain (captured by Xception + 2D FFT).

**Q2: What do APCER, BPCER, and ACER measure under ISO/IEC 30107-3?**
**A:**
- **APCER (Attack Presentation Classification Error Rate):** Proportion of spoof attacks mistakenly accepted as genuine (security failure rate).
- **BPCER (Bona Fide Presentation Classification Error Rate):** Proportion of genuine users mistakenly rejected as spoofs (user friction rate).
- **ACER (Average Classification Error Rate):** The arithmetic mean `(APCER + BPCER) / 2`.

**Q3: Why is random frame splitting invalid when training biometric spoof detectors?**
**A:** If frames from the same person or video appear in both training and test sets, the CNN memorizes subject identity, lighting, or background rather than spoofing artifacts, inflating test accuracy. Grouping splits strictly by `subject_id` prevents identity leakage.

**Q4: Does passing this anti-spoofing check prove the user is the legitimate bank account owner?**
**A:** No. Anti-spoofing verifies only that the biometric sample appears to come from a live, unmanipulated human face rather than a photo, screen, or deepfake. It must be paired with an identity matcher (e.g., ArcFace/FaceNet) to confirm who the person is.
