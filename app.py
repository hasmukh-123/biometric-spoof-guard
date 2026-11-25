"""
AI-Based Physical and Digital Biometric Spoofing Detection for Secure E-Transactions
Main Streamlit Frontend Application (Zero OpenCV DLL dependency).
"""
import io
import json
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

import config
from inference.predictor import BiometricSpoofPredictor, PredictionReport
from utils.security import validate_uploaded_file, compute_sha256
from utils.database import log_detection_event, fetch_recent_logs, clear_logs

st.set_page_config(
    page_title="BioGuard AI | Biometric Spoofing Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Modern Cybersecurity Styling
st.markdown(
    """
    <style>
    .main-header {
        background: linear-gradient(90deg, #0f172a 0%, #1e3a8a 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 12px;
        color: #f8fafc;
        margin-bottom: 1.2rem;
    }
    .main-header h1 { margin: 0; font-size: 1.75rem; font-weight: 700; }
    .main-header p { margin: 0.35rem 0 0 0; color: #cbd5e1; font-size: 0.95rem; }
    .verdict-box {
        padding: 1.1rem 1.3rem;
        border-radius: 10px;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .verdict-low { background-color: #ecfdf5; border-left: 6px solid #10b981; color: #065f46; }
    .verdict-high { background-color: #fef2f2; border-left: 6px solid #ef4444; color: #991b1b; }
    .verdict-warn { background-color: #fffbeb; border-left: 6px solid #f59e0b; color: #92400e; }
    .privacy-banner {
        background-color: #f1f5f9;
        border: 1px solid #cbd5e1;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        font-size: 0.85rem;
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_predictor() -> BiometricSpoofPredictor:
    return BiometricSpoofPredictor()


def render_prediction_results(report: PredictionReport) -> None:
    """Render standardized prediction verdict, confidence gauges, forensic maps, and security disclaimer."""
    css_class = {
        "LOW": "verdict-low",
        "HIGH": "verdict-high",
        "WARNING": "verdict-warn",
    }.get(report.risk_level, "verdict-warn")

    st.markdown(
        f"""
        <div class="verdict-box {css_class}">
            <div style="font-size:1.2rem;">Decision: {report.verdict}</div>
            <div style="font-size:0.9rem; margin-top:4px;">
                Decision Confidence: <b>{report.confidence_score * 100:.1f}%</b> &nbsp;|&nbsp;
                Inference Latency: <b>{report.latency_ms:.1f} ms</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if report.risk_level == "HIGH":
        st.error(
            "**SECURITY WARNING — E-TRANSACTION BLOCKED:** Suspected biometric spoofing or manipulation detected. "
            "Step-up authentication (hardware security key / OTP + manual review) is required."
        )
    elif report.risk_level == "WARNING":
        st.warning(
            "**VERIFICATION INCONCLUSIVE:** Input could not be cleared with high confidence. "
            "Please re-capture a well-lit frontal face image."
        )
    else:
        st.info(
            "**Security Layer Notice:** This check confirms only that no physical or digital spoofing cues were "
            "detected. Liveness/spoof detection is an **anti-spoofing defense layer**, not standalone identity "
            "verification—it must be combined with facial recognition template matching."
        )

    # Probability Metrics Row
    col1, col2, col3 = st.columns(3)
    with col1:
        pad_display = f"{report.physical_spoof_prob * 100:.1f}%" if report.physical_spoof_prob is not None else "N/A"
        st.metric("Physical Spoof Risk (Model A)", pad_display)
        if report.physical_spoof_prob is not None:
            st.progress(min(1.0, max(0.0, report.physical_spoof_prob)))
    with col2:
        df_display = f"{report.digital_deepfake_prob * 100:.1f}%" if report.digital_deepfake_prob is not None else "N/A"
        st.metric("Digital Deepfake Risk (Model B)", df_display)
        if report.digital_deepfake_prob is not None:
            st.progress(min(1.0, max(0.0, report.digital_deepfake_prob)))
    with col3:
        st.metric("Overall Decision Confidence", f"{report.confidence_score * 100:.1f}%")
        st.progress(min(1.0, max(0.0, report.confidence_score)))

    # Forensic Explanation
    st.subheader("Diagnostic Findings & Forensic Explanations")
    for note in report.explanations:
        st.write(f"- {note}")

    if report.forensic_signals:
        with st.expander("View Quantitative Signal Features (LBP, Specular Glare, 2D FFT)"):
            sig_df = pd.DataFrame([report.forensic_signals])
            st.dataframe(sig_df, use_container_width=True)

    # Visual Diagnostics
    if report.annotated_frame_rgb is not None:
        st.subheader("Computer Vision Visual Diagnostics")
        v1, v2, v3 = st.columns(3)
        with v1:
            st.image(report.annotated_frame_rgb, caption="Detected Face ROI", use_container_width=True)
        with v2:
            if report.lbp_visual is not None:
                st.image(report.lbp_visual, caption="LBP Micro-Texture Map (Physical PAD)", use_container_width=True)
        with v3:
            if report.fft_visual is not None:
                st.image(report.fft_visual, caption="2D FFT Frequency Spectrum (Deepfake)", use_container_width=True)


def main() -> None:
    predictor = load_predictor()

    st.markdown(
        """
        <div class="main-header">
            <h1>🛡️ BioGuard AI — Biometric Spoofing Detection for Secure E-Transactions</h1>
            <p>Dual-Stream Physical Presentation Attack Detection (PAD) & Digital Deepfake Defense System</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sidebar Controls & Operational Status
    with st.sidebar:
        st.header("Navigation & Controls")
        page = st.radio(
            "Select Module",
            [
                "Home Dashboard",
                "Image Spoof Analysis",
                "Video Spoof Analysis",
                "Live Webcam Check",
                "Model Performance & Evaluation",
                "Privacy & Audit History",
                "About Project",
            ],
        )
        st.divider()
        detection_mode = st.selectbox(
            "Detection Pipeline Mode",
            [
                "Both (Full Security Scan)",
                "Physical PAD Only",
                "Digital Deepfake Only",
            ],
        )
        st.divider()
        st.subheader("Model Weight Status")
        if predictor.pad_trained:
            st.success("Model A (Physical PAD): Fine-Tuned Weights Active")
        else:
            st.warning("Model A (Physical PAD): Heuristic Signal Mode (Awaiting Dataset Training)")

        if predictor.deepfake_trained:
            st.success("Model B (Digital Deepfake): Fine-Tuned Weights Active")
        else:
            st.warning("Model B (Digital Deepfake): Heuristic Signal Mode (Awaiting Dataset Training)")

        st.markdown(
            """
            <div class="privacy-banner">
                <b>Privacy Guarantee:</b> Uploaded faces and webcam frames are processed strictly in volatile RAM
                and immediately discarded. Only SHA-256 hashes and risk scores are logged.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------------- PAGE 1: HOME DASHBOARD ----------------
    if page == "Home Dashboard":
        st.subheader("E-Transaction Biometric Threat Defense Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Physical PAD Backbone", "EfficientNet-B0 + LBP")
        c2.metric("Deepfake Backbone", "Xception + 2D FFT")
        c3.metric("PAD Threshold", f"{config.PHYSICAL_SPOOF_THRESHOLD:.2f}")
        c4.metric("Deepfake Threshold", f"{config.DIGITAL_DEEPFAKE_THRESHOLD:.2f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### Physical Presentation Attack Detection (Model A)")
            st.write(
                "- **Target Threats:** Printed high-resolution photos, smartphone/tablet screen replay attacks, "
                "and recorded video playbacks held before a KYC or payment camera.\n"
                "- **Signal Extraction:** Combines RGB spatial features (`EfficientNet-B0`) with Local Binary "
                "Pattern (`LBP`) micro-texture maps and specular screen glare detection.\n"
                "- **ISO/IEC 30107-3 Metrics:** Evaluated via **APCER**, **BPCER**, and **ACER**."
            )
        with col_b:
            st.markdown("### Digital Deepfake Detection (Model B)")
            st.write(
                "- **Target Threats:** Real-time virtual camera face swaps, GAN/diffusion synthetic faces, "
                "and manipulated facial videos injected into digital banking sessions.\n"
                "- **Signal Extraction:** Combines `Xception` spatial boundary features with centered **2D FFT** "
                "log-magnitude spectra to expose upsampling frequency artifacts.\n"
                "- **Classification Metrics:** Evaluated via **Precision**, **Recall**, **F1-Score**, and **ROC-AUC**."
            )

    # ---------------- PAGE 2: IMAGE UPLOAD ----------------
    elif page == "Image Spoof Analysis":
        st.subheader("Facial Image Spoofing & Manipulation Analysis")
        uploaded_img = st.file_uploader(
            "Upload a facial image (JPG, PNG, WEBP — Max 10 MB)",
            type=["jpg", "jpeg", "png", "webp"],
        )
        if uploaded_img is not None:
            raw_bytes = uploaded_img.getvalue()
            is_valid, msg = validate_uploaded_file(uploaded_img.name, raw_bytes, expected_kind="image")
            if not is_valid:
                st.error(f"Input Validation Rejected: {msg}")
            else:
                try:
                    image_rgb = np.asarray(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
                except Exception:
                    image_rgb = None

                if image_rgb is None:
                    st.error("Corrupted image file could not be decoded.")
                else:
                    st.image(image_rgb, caption="Uploaded Image", width=320)
                    if st.button("Analyze Facial Image", type="primary"):
                        with st.spinner("Extracting face ROI, LBP texture, and 2D FFT spectral cues..."):
                            report = predictor.analyze_image(image_rgb, mode=detection_mode)
                            log_detection_event(
                                {
                                    "input_sha256": compute_sha256(raw_bytes),
                                    "media_type": "image",
                                    "detection_mode": detection_mode,
                                    "model_status": report.model_status,
                                    "physical_spoof_prob": report.physical_spoof_prob,
                                    "digital_deepfake_prob": report.digital_deepfake_prob,
                                    "verdict": report.verdict,
                                    "latency_ms": report.latency_ms,
                                }
                            )
                        render_prediction_results(report)

    # ---------------- PAGE 3: VIDEO UPLOAD ----------------
    elif page == "Video Spoof Analysis":
        st.subheader("Facial Video Replay & Deepfake Video Analysis")
        uploaded_vid = st.file_uploader(
            "Upload a facial verification video (MP4, AVI, MOV, MKV — Max 50 MB)",
            type=["mp4", "avi", "mov", "mkv"],
        )
        if uploaded_vid is not None:
            raw_bytes = uploaded_vid.getvalue()
            is_valid, msg = validate_uploaded_file(uploaded_vid.name, raw_bytes, expected_kind="video")
            if not is_valid:
                st.error(f"Input Validation Rejected: {msg}")
            else:
                st.video(raw_bytes)
                if st.button("Analyze Video Frames", type="primary"):
                    with st.spinner(f"Sampling {config.VIDEO_SAMPLE_FRAMES} frames and aggregating risk scores..."):
                        suffix = "." + uploaded_vid.name.split(".")[-1].lower()
                        report = predictor.analyze_video(raw_bytes, suffix=suffix, mode=detection_mode)
                        log_detection_event(
                            {
                                "input_sha256": compute_sha256(raw_bytes),
                                "media_type": "video",
                                "detection_mode": detection_mode,
                                "model_status": report.model_status,
                                "physical_spoof_prob": report.physical_spoof_prob,
                                "digital_deepfake_prob": report.digital_deepfake_prob,
                                "verdict": report.verdict,
                                "latency_ms": report.latency_ms,
                            }
                        )
                    render_prediction_results(report)

    # ---------------- PAGE 4: LIVE WEBCAM ----------------
    elif page == "Live Webcam Check":
        st.subheader("Live Webcam Liveness & Anti-Spoofing Capture")
        st.write("Capture a live frame from your device camera to simulate an e-transaction biometric check.")
        cam_photo = st.camera_input("Capture Facial Frame")
        if cam_photo is not None:
            raw_bytes = cam_photo.getvalue()
            try:
                image_rgb = np.asarray(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
            except Exception:
                image_rgb = None

            if image_rgb is not None and st.button("Analyze Webcam Frame", type="primary"):
                with st.spinner("Running real-time anti-spoofing inspection..."):
                    report = predictor.analyze_image(image_rgb, mode=detection_mode)
                    log_detection_event(
                        {
                            "input_sha256": compute_sha256(raw_bytes),
                            "media_type": "webcam",
                            "detection_mode": detection_mode,
                            "model_status": report.model_status,
                            "physical_spoof_prob": report.physical_spoof_prob,
                            "digital_deepfake_prob": report.digital_deepfake_prob,
                            "verdict": report.verdict,
                            "latency_ms": report.latency_ms,
                        }
                    )
                render_prediction_results(report)

    # ---------------- PAGE 5: MODEL PERFORMANCE & EVALUATION ----------------
    elif page == "Model Performance & Evaluation":
        st.subheader("Model Evaluation & ISO/IEC 30107-3 Benchmark Reports")
        st.write(
            "This dashboard displays genuine evaluation outputs generated by `evaluation/evaluate_models.py` "
            "after training on subject-disjoint splits. No fabricated accuracy numbers are displayed."
        )

        for task_key, title in [
            ("pad", "Model A: Physical Presentation Attack Detection (EfficientNet-B0 + LBP)"),
            ("deepfake", "Model B: Digital Deepfake Detection (Xception + 2D FFT)"),
        ]:
            st.markdown(f"### {title}")
            report_file = config.REPORTS_DIR / f"{task_key}_evaluation_report.json"
            cm_file = config.REPORTS_DIR / f"{task_key}_confusion_matrix.png"

            if report_file.exists():
                with open(report_file, "r", encoding="utf-8") as f:
                    metrics = json.load(f)
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
                m2.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                m3.metric("APCER", f"{metrics['APCER']:.4f}")
                m4.metric("BPCER", f"{metrics['BPCER']:.4f}")
                m5.metric("ACER", f"{metrics['ACER']:.4f}")

                if cm_file.exists():
                    st.image(str(cm_file), width=420)
            else:
                st.info(
                    f"No evaluation report generated yet for `{task_key}`. "
                    f"Once you integrate the dataset and run `python -m training.train_{task_key}` followed by "
                    f"`python -m evaluation.evaluate_models --task {task_key}`, verified metrics will appear here."
                )

    # ---------------- PAGE 6: PRIVACY & AUDIT HISTORY ----------------
    elif page == "Privacy & Audit History":
        st.subheader("Privacy-Preserving Audit Log (Zero Biometric Image Storage)")
        st.write(
            "In compliance with biometric data minimization principles, only cryptographic SHA-256 hashes, "
            "timestamps, model modes, and risk scores are persisted in SQLite."
        )
        logs = fetch_recent_logs(limit=100)
        if logs:
            df_logs = pd.DataFrame(logs)
            st.dataframe(df_logs, use_container_width=True)
            if st.button("Purge Audit Logs"):
                clear_logs()
                st.success("Audit history cleared.")
                st.rerun()
        else:
            st.info("No detection events logged yet.")

    # ---------------- PAGE 7: ABOUT PROJECT ----------------
    elif page == "About Project":
        st.subheader("Seminar Project: Biometrics in Secure E-Transactions")
        st.markdown(
            """
            **System Purpose:** High-value digital banking, remote e-KYC, and account recovery workflows increasingly rely
            on facial biometrics. Attackers bypass standard face recognition using either **Physical Presentation Attacks**
            (printed photos, high-DPI tablet replays) or **Digital Injection Attacks** (real-time face swaps and deepfakes).

            **Key Architectural Safeguards:**
            1. **Dual-Stream Detection:** Separates physical surface/glare artifacts from digital frequency/blending artifacts.
            2. **Leakage-Free Evaluation:** Uses `GroupShuffleSplit` on `subject_id` so identities never overlap across splits.
            3. **Honest Decision Semantics:** Distinguishes *Genuine-Looking*, *Suspected Physical Spoof*,
               *Suspected Digital Manipulation*, and *Uncertain Input* without overclaiming identity verification.
            """
        )


if __name__ == "__main__":
    main()
