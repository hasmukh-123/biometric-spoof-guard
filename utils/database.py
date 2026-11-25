"""
Privacy-conscious SQLite audit logger.
Stores ONLY cryptographic hashes, timestamps, risk scores, and decision outcomes.
Raw facial images and videos are processed in RAM and immediately discarded.
"""
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Any
import config


def init_db() -> None:
    """Create the audit_logs table if it does not exist."""
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc TEXT NOT NULL,
                input_sha256 TEXT NOT NULL,
                media_type TEXT NOT NULL,
                detection_mode TEXT NOT NULL,
                model_status TEXT NOT NULL,
                physical_spoof_prob REAL,
                digital_deepfake_prob REAL,
                verdict TEXT NOT NULL,
                latency_ms REAL NOT NULL
            )
            """
        )
        conn.commit()


def log_detection_event(record: Dict[str, Any]) -> None:
    """Insert a single anonymized detection audit event."""
    init_db()
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO audit_logs (
                timestamp_utc, input_sha256, media_type, detection_mode,
                model_status, physical_spoof_prob, digital_deepfake_prob,
                verdict, latency_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                record["input_sha256"],
                record["media_type"],
                record["detection_mode"],
                record["model_status"],
                record.get("physical_spoof_prob"),
                record.get("digital_deepfake_prob"),
                record["verdict"],
                round(record["latency_ms"], 2),
            ),
        )
        conn.commit()


def fetch_recent_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent audit logs without exposing any biometric pixels."""
    init_db()
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def clear_logs() -> None:
    """Purge all audit history."""
    init_db()
    with sqlite3.connect(config.DB_PATH) as conn:
        conn.execute("DELETE FROM audit_logs")
        conn.commit()
