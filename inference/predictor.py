"""
Unified Biometric Spoofing Inference Engine (Zero OpenCV DLL dependency).
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from PIL import Image, ImageDraw
import torch

import config
from preprocessing.face_detector import FacePreprocessor
from preprocessing.transforms import (
    compute_lbp_map,
    compute_fft_spectrum,
    prepare_pad_tensors,
    prepare_deepfake_tensors,
)
from models.physical_pad_model import PhysicalPADNet
from models.digital_deepfake_model import DigitalDeepfakeNet


@dataclass
class PredictionReport:
    verdict: str
    risk_level: str
    physical_spoof_prob: Optional[float]
    digital_deepfake_prob: Optional[float]
    confidence_score: float
    model_status: str
    explanations: List[str] = field(default_factory=list)
    forensic_signals: Dict[str, float] = field(default_factory=dict)
    annotated_frame_rgb: Optional[np.ndarray] = None
    lbp_visual: Optional[np.ndarray] = None
    fft_visual: Optional[np.ndarray] = None
    latency_ms: float = 0.0


class BiometricSpoofPredictor:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.preprocessor = FacePreprocessor()

        self.pad_model: Optional[PhysicalPADNet] = None
        self.pad_trained: bool = False
        self.deepfake_model: Optional[DigitalDeepfakeNet] = None
        self.deepfake_trained: bool = False

        self._load_available_weights()

    def _load_available_weights(self) -> None:
        if config.PAD_MODEL_PATH.exists():
            try:
                model = PhysicalPADNet(pretrained_backbone=False).to(self.device)
                ckpt = torch.load(config.PAD_MODEL_PATH, map_location=self.device)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                self.pad_model = model
                self.pad_trained = True
            except Exception as exc:
                print(f"[WARN] Could not load PAD weights ({exc}); using forensic signal mode.")

        if config.DEEPFAKE_MODEL_PATH.exists():
            try:
                model = DigitalDeepfakeNet(pretrained_backbone=False).to(self.device)
                ckpt = torch.load(config.DEEPFAKE_MODEL_PATH, map_location=self.device)
                model.load_state_dict(ckpt["model_state_dict"])
                model.eval()
                self.deepfake_model = model
                self.deepfake_trained = True
            except Exception as exc:
                print(f"[WARN] Could not load Deepfake weights ({exc}); using forensic signal mode.")

    def get_model_status_summary(self) -> str:
        pad_state = "Trained Checkpoint Loaded" if self.pad_trained else "Heuristic Signal Mode (Weights Pending)"
        df_state = "Trained Checkpoint Loaded" if self.deepfake_trained else "Heuristic Signal Mode (Weights Pending)"
        return f"Model A (PAD): {pad_state} | Model B (Deepfake): {df_state}"

    @staticmethod
    def _extract_forensic_cues(face_rgb: np.ndarray) -> Dict[str, float]:
        lbp = compute_lbp_map(face_rgb)
        fft_spec = compute_fft_spectrum(face_rgb)

        hist, _ = np.histogram(lbp, bins=32, range=(0.0, 1.0), density=True)
        hist = hist[hist > 0]
        lbp_entropy = float(-np.sum(hist * np.log2(hist + 1e-9)) / 5.0)

        # Specular glare via HSV Value & Saturation
        rgb_norm = face_rgb.astype(np.float32) / 255.0
        cmax = np.max(rgb_norm, axis=2)
        cmin = np.min(rgb_norm, axis=2)
        sat = np.where(cmax > 0, (cmax - cmin) / (cmax + 1e-6), 0.0)
        specular_ratio = float(np.mean((cmax > 0.95) & (sat < 0.14)))

        h, w = fft_spec.shape
        cy, cx = h // 2, w // 2
        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((y_grid - cy) ** 2 + (x_grid - cx) ** 2)
        outer_mask = dist > (0.35 * min(h, w))
        high_freq_energy = float(np.mean(fft_spec[outer_mask]))

        # Chrominance variance in Cb/Cr
        r, g, b = face_rgb[:, :, 0].astype(np.float32), face_rgb[:, :, 1].astype(np.float32), face_rgb[:, :, 2].astype(np.float32)
        cb = -0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 * r - 0.418688 * g - 0.081312 * b
        chrominance_std = float(0.5 * (np.std(cb) + np.std(cr)) / 40.0)

        return {
            "lbp_texture_score": round(float(np.clip(lbp_entropy, 0.0, 1.0)), 4),
            "specular_glare_ratio": round(float(np.clip(specular_ratio, 0.0, 1.0)), 4),
            "fft_high_freq_energy": round(float(np.clip(high_freq_energy, 0.0, 1.0)), 4),
            "chrominance_richness": round(float(np.clip(chrominance_std, 0.0, 1.0)), 4),
        }

    def _predict_single_face(
        self, face_rgb: np.ndarray, cues: Dict[str, float], mode: str
    ) -> tuple[Optional[float], Optional[float], List[str]]:
        explanations: List[str] = []
        pad_prob: Optional[float] = None
        df_prob: Optional[float] = None

        if mode in {"Both (Full Security Scan)", "Physical PAD Only"}:
            if self.pad_trained and self.pad_model is not None:
                rgb_t, lbp_t = prepare_pad_tensors(face_rgb, is_train=False)
                with torch.no_grad():
                    logit = self.pad_model(rgb_t.unsqueeze(0).to(self.device), lbp_t.unsqueeze(0).to(self.device))
                    pad_prob = float(torch.sigmoid(logit).item())
            else:
                glare_risk = min(1.0, cues["specular_glare_ratio"] * 4.5)
                flat_color_risk = max(0.0, 1.0 - cues["chrominance_richness"])
                flat_texture_risk = max(0.0, 0.85 - cues["lbp_texture_score"])
                pad_prob = float(np.clip(0.45 * glare_risk + 0.30 * flat_color_risk + 0.25 * flat_texture_risk, 0.05, 0.95))

            if pad_prob >= config.PHYSICAL_SPOOF_THRESHOLD:
                explanations.append(
                    f"Physical PAD Alert (score={pad_prob:.2f}): Elevated screen specular glare "
                    f"({cues['specular_glare_ratio']:.3f}) or flattened print/display micro-texture."
                )

        if mode in {"Both (Full Security Scan)", "Digital Deepfake Only"}:
            if self.deepfake_trained and self.deepfake_model is not None:
                rgb_t, fft_t = prepare_deepfake_tensors(face_rgb, is_train=False)
                with torch.no_grad():
                    logit = self.deepfake_model(rgb_t.unsqueeze(0).to(self.device), fft_t.unsqueeze(0).to(self.device))
                    df_prob = float(torch.sigmoid(logit).item())
            else:
                freq_anomaly = abs(cues["fft_high_freq_energy"] - 0.42) * 1.8
                over_smooth_risk = max(0.0, 0.80 - cues["lbp_texture_score"])
                df_prob = float(np.clip(0.55 * freq_anomaly + 0.45 * over_smooth_risk, 0.05, 0.95))

            if df_prob >= config.DIGITAL_DEEPFAKE_THRESHOLD:
                explanations.append(
                    f"Digital Manipulation Alert (score={df_prob:.2f}): Abnormal 2D FFT spectral energy "
                    f"({cues['fft_high_freq_energy']:.3f}) or synthetic skin over-smoothing."
                )

        return pad_prob, df_prob, explanations

    def analyze_image(
        self, image_rgb: np.ndarray, mode: str = "Both (Full Security Scan)"
    ) -> PredictionReport:
        start_t = time.perf_counter()
        face_res = self.preprocessor.extract_primary_face(image_rgb)

        if not face_res.success or face_res.face_rgb is None:
            latency_ms = (time.perf_counter() - start_t) * 1000.0
            return PredictionReport(
                verdict="Uncertain or Unsupported Input",
                risk_level="WARNING",
                physical_spoof_prob=None,
                digital_deepfake_prob=None,
                confidence_score=0.0,
                model_status=self.get_model_status_summary(),
                explanations=[face_res.message],
                latency_ms=latency_ms,
            )

        cues = self._extract_forensic_cues(face_res.face_rgb)
        pad_prob, df_prob, explanations = self._predict_single_face(face_res.face_rgb, cues, mode)

        # Draw bounding box using PIL
        pil_annotated = Image.fromarray(image_rgb.astype(np.uint8)).copy()
        if face_res.bbox is not None:
            x, y, w, h = face_res.bbox
            draw = ImageDraw.Draw(pil_annotated)
            draw.rectangle([x, y, x + w, y + h], outline=(0, 210, 255), width=3)
        annotated_rgb = np.asarray(pil_annotated)

        lbp_vis = (compute_lbp_map(face_res.face_rgb) * 255).astype(np.uint8)
        fft_vis = (compute_fft_spectrum(face_res.face_rgb) * 255).astype(np.uint8)

        verdict, risk_level, confidence, fusion_notes = self._fuse_decision(
            pad_prob, df_prob, face_res.quality_ok, face_res.message
        )
        explanations.extend(fusion_notes)

        latency_ms = (time.perf_counter() - start_t) * 1000.0
        return PredictionReport(
            verdict=verdict,
            risk_level=risk_level,
            physical_spoof_prob=round(pad_prob, 4) if pad_prob is not None else None,
            digital_deepfake_prob=round(df_prob, 4) if df_prob is not None else None,
            confidence_score=round(confidence, 4),
            model_status=self.get_model_status_summary(),
            explanations=explanations,
            forensic_signals=cues,
            annotated_frame_rgb=annotated_rgb,
            lbp_visual=lbp_vis,
            fft_visual=fft_vis,
            latency_ms=latency_ms,
        )

    def analyze_video(
        self, video_bytes: bytes, suffix: str = ".mp4", mode: str = "Both (Full Security Scan)"
    ) -> PredictionReport:
        start_t = time.perf_counter()
        frames = self.preprocessor.sample_video_frames(video_bytes, suffix=suffix)

        if not frames:
            return PredictionReport(
                verdict="Uncertain or Unsupported Input",
                risk_level="WARNING",
                physical_spoof_prob=None,
                digital_deepfake_prob=None,
                confidence_score=0.0,
                model_status=self.get_model_status_summary(),
                explanations=["Could not decode readable video frames from uploaded file."],
                latency_ms=(time.perf_counter() - start_t) * 1000.0,
            )

        pad_scores: List[float] = []
        df_scores: List[float] = []
        valid_frames = 0
        representative_report: Optional[PredictionReport] = None

        for frame_rgb in frames:
            rep = self.analyze_image(frame_rgb, mode=mode)
            if rep.verdict != "Uncertain or Unsupported Input" or (
                rep.physical_spoof_prob is not None or rep.digital_deepfake_prob is not None
            ):
                valid_frames += 1
                if representative_report is None:
                    representative_report = rep
                if rep.physical_spoof_prob is not None:
                    pad_scores.append(rep.physical_spoof_prob)
                if rep.digital_deepfake_prob is not None:
                    df_scores.append(rep.digital_deepfake_prob)

        if valid_frames == 0 or representative_report is None:
            return PredictionReport(
                verdict="Uncertain or Unsupported Input",
                risk_level="WARNING",
                physical_spoof_prob=None,
                digital_deepfake_prob=None,
                confidence_score=0.0,
                model_status=self.get_model_status_summary(),
                explanations=[f"No clear frontal face detected across {len(frames)} sampled video frames."],
                latency_ms=(time.perf_counter() - start_t) * 1000.0,
            )

        agg_pad = float(np.percentile(pad_scores, 75)) if pad_scores else None
        agg_df = float(np.percentile(df_scores, 75)) if df_scores else None

        verdict, risk_level, confidence, notes = self._fuse_decision(
            agg_pad, agg_df, quality_ok=True, quality_msg=f"Analyzed {valid_frames}/{len(frames)} valid face frames."
        )
        notes.insert(0, f"Video Temporal Aggregation: Evaluated {valid_frames} valid face frames out of {len(frames)} sampled.")

        return PredictionReport(
            verdict=verdict,
            risk_level=risk_level,
            physical_spoof_prob=round(agg_pad, 4) if agg_pad is not None else None,
            digital_deepfake_prob=round(agg_df, 4) if agg_df is not None else None,
            confidence_score=round(confidence, 4),
            model_status=self.get_model_status_summary(),
            explanations=notes,
            forensic_signals=representative_report.forensic_signals,
            annotated_frame_rgb=representative_report.annotated_frame_rgb,
            lbp_visual=representative_report.lbp_visual,
            fft_visual=representative_report.fft_visual,
            latency_ms=(time.perf_counter() - start_t) * 1000.0,
        )

    @staticmethod
    def _fuse_decision(
        pad_prob: Optional[float],
        df_prob: Optional[float],
        quality_ok: bool,
        quality_msg: str,
    ) -> tuple[str, str, float, List[str]]:
        notes: List[str] = []
        if not quality_ok:
            notes.append(quality_msg)

        pad_hit = pad_prob is not None and pad_prob >= config.PHYSICAL_SPOOF_THRESHOLD
        df_hit = df_prob is not None and df_prob >= config.DIGITAL_DEEPFAKE_THRESHOLD

        if pad_hit and (not df_hit or (pad_prob or 0.0) >= (df_prob or 0.0)):
            assert pad_prob is not None
            notes.append("High physical presentation attack risk detected (e.g., printed photo or screen replay).")
            return "Suspected Physical Spoof", "HIGH", pad_prob, notes

        if df_hit:
            assert df_prob is not None
            notes.append("High digital manipulation risk detected (e.g., face-swap or synthetic deepfake).")
            return "Suspected Digital Manipulation", "HIGH", df_prob, notes

        near_pad = pad_prob is not None and abs(pad_prob - config.PHYSICAL_SPOOF_THRESHOLD) <= config.UNCERTAIN_MARGIN
        near_df = df_prob is not None and abs(df_prob - config.DIGITAL_DEEPFAKE_THRESHOLD) <= config.UNCERTAIN_MARGIN

        if not quality_ok or near_pad or near_df:
            max_risk = max([p for p in (pad_prob, df_prob) if p is not None], default=0.5)
            notes.append("Borderline risk score or degraded visual quality; request re-capture under clear lighting.")
            return "Uncertain or Unsupported Input", "WARNING", 1.0 - abs(0.5 - max_risk), notes

        active_probs = [p for p in (pad_prob, df_prob) if p is not None]
        benign_conf = 1.0 - max(active_probs) if active_probs else 0.5
        notes.append(
            "No physical presentation or digital manipulation artifacts detected above risk thresholds. "
            "Note: Liveness/spoof clearance confirms input integrity only and must be paired with identity matching."
        )
        return "Genuine-Looking Face Input", "LOW", benign_conf, notes