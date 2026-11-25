# Seminar Report: AI-Based Physical and Digital Biometric Spoofing Detection for Secure E-Transactions

## Abstract
Remote financial onboarding (e-KYC), mobile banking, and high-value e-transactions rely heavily on facial biometrics. However, traditional facial recognition systems verify *identity similarity* without verifying *physical presence* or *digital signal integrity*. This project presents **BioGuard AI**, a dual-stream anti-spoofing system combining a **Physical Presentation Attack Detection (PAD)** pipeline (`EfficientNet-B0` + Local Binary Pattern micro-texture maps) with a **Digital Deepfake Detection** pipeline (`Xception` + 2D Fast Fourier Transform frequency spectra).

## 1. Threat Model in E-Transactions
1. **Physical Presentation Attacks (ISO/IEC 30107-1):**
   - *Print Attacks:* High-resolution photographs printed on matte or glossy paper.
   - *Replay Attacks:* Recorded facial videos played back on OLED/LCD smartphone or tablet displays.
2. **Digital Injection Attacks:**
   - *Face Swapping & Reenactment:* Real-time virtual camera feeds injecting swapped identities.
   - *Synthetic GAN/Diffusion Faces:* Deepfake faces generated to bypass remote KYC checks.

## 2. System Methodology
- **Preprocessing & Quality Gate:** OpenCV face detection with a 22% contextual margin captures screen bezels and jawline blending boundaries, while Laplacian variance filtering rejects severely blurred inputs as *Uncertain*.
- **Model A (Physical PAD):** Fuses 1280-D `EfficientNet-B0` embeddings with a 32-D convolutional branch over 8-neighbor `LBP` maps to detect paper grain and display subpixel periodicity.
- **Model B (Digital Deepfake):** Fuses 2048-D `Xception` embeddings with a 32-D convolutional branch over centered `2D FFT` log-magnitude spectra to expose generative upsampling artifacts.
- **Leakage-Free Validation:** `GroupShuffleSplit` grouped by `subject_id` guarantees zero subject or source-video overlap across training, validation, and test partitions.
- **Privacy Architecture:** Facial frames are processed strictly in volatile memory; only SHA-256 digests and risk scores are persisted in SQLite.

## 3. Security Scope & Limitations
Anti-spoofing confirms **capture integrity** (*is this a live, unmanipulated human presentation?*), not **claimant identity** (*is this the authorized account holder?*). Production e-transaction systems must execute BioGuard AI as a mandatory pre-filter ahead of deep facial embedding verification (e.g., ArcFace) and transaction risk scoring.
