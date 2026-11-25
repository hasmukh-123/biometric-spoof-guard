# Seminar Presentation Content: Biometrics in Secure E-Transactions

## Slide 1: Title Slide
- **Title:** AI-Based Physical and Digital Biometric Spoofing Detection for Secure E-Transactions
- **Domain:** Biometric Cybersecurity, Deep Learning & Computer Vision

## Slide 2: Problem Statement — The E-Transaction Vulnerability
- Remote e-KYC and mobile payments trust facial cameras as proof of presence.
- **Physical Spoofs:** Printed photos and smartphone screen replays fool standard face matchers.
- **Digital Spoofs:** Virtual cameras and real-time deepfakes bypass physical camera optics entirely.

## Slide 3: Dual-Stream System Architecture
- **Gateway:** Magic-byte validation + Face ROI extraction + Laplacian blur quality gate.
- **Stream A (Physical PAD):** `EfficientNet-B0` + `LBP` Micro-Texture Branch.
- **Stream B (Digital Deepfake):** `Xception` + `2D FFT` Frequency Spectrum Branch.
- **Risk Fusion:** Outputs 4 states (*Genuine-Looking*, *Suspected Physical Spoof*, *Suspected Digital Manipulation*, *Uncertain*).

## Slide 4: ISO/IEC 30107-3 & Deepfake Evaluation Methodology
- **Zero Data Leakage:** Subject/video-disjoint splitting via `GroupShuffleSplit`.
- **PAD Metrics:** APCER (attack acceptance), BPCER (false rejection), ACER (balanced error).
- **Deepfake Metrics:** Precision, Recall, F1-Score, ROC-AUC.

## Slide 5: Privacy-by-Design & Security Controls
- Strict file-type/magic-byte verification and size caps (10 MB image / 50 MB video).
- Zero biometric pixel storage: volatile RAM processing + SHA-256 audit logging in SQLite.
- Explicit separation between **Anti-Spoofing Clearance** and **Identity Verification**.
