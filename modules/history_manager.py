"""
history_manager.py - Local CSV detection history.

Privacy: only a SANITISED URL is stored. Any username:password before '@' is removed
and query-string values are replaced with '***' (they often contain tokens or personal
data). Passwords or personal information are never written to disk.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from urllib.parse import urlsplit, urlunsplit

import pandas as pd

import config

COLUMNS = ["timestamp", "url", "final_score", "risk_category", "ml_prediction",
           "rule_score", "ml_score", "indicator_count", "source"]


def sanitize_url_for_storage(url: str) -> str:
    """Strip credentials and query values before saving a URL."""
    try:
        parts = urlsplit(url.strip())
        netloc = parts.netloc.rpartition("@")[2] if "@" in parts.netloc else parts.netloc
        query = "&".join(f"{pair.split('=', 1)[0]}=***" if "=" in pair else pair
                         for pair in parts.query.split("&") if pair)
        clean = urlunsplit((parts.scheme, netloc, parts.path, query, ""))
        return clean[:500]
    except ValueError:
        return url.strip()[:500]


def add_record(assessment, source: str = "manual", path: Path = config.HISTORY_PATH) -> Dict:
    """Append one analysis result (a RiskAssessment) to the history file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "url": sanitize_url_for_storage(assessment.url),
        "final_score": assessment.final_score,
        "risk_category": assessment.category,
        "ml_prediction": assessment.ml_label or "N/A",
        "rule_score": assessment.rule_score,
        "ml_score": assessment.ml_score if assessment.ml_score is not None else "",
        "indicator_count": len(assessment.indicators),
        "source": source,
    }
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(record)
    return record


def load_history(path: Path = config.HISTORY_PATH) -> pd.DataFrame:
    """Return the history as a DataFrame (newest first). Empty frame if none/corrupt."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=COLUMNS)
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=COLUMNS)
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNS].iloc[::-1].reset_index(drop=True)


def clear_history(path: Path = config.HISTORY_PATH) -> None:
    """Delete the local history file."""
    Path(path).unlink(missing_ok=True)


def get_statistics(path: Path = config.HISTORY_PATH) -> Dict:
    """Counts used by the dashboard. SAFE/SUSPICIOUS/MALICIOUS map to LOW/MEDIUM/HIGH RISK."""
    df = load_history(path)
    counts = df["risk_category"].value_counts().to_dict() if len(df) else {}
    return {
        "total": int(len(df)),
        "safe": int(counts.get("LOW RISK", 0)),
        "suspicious": int(counts.get("MEDIUM RISK", 0)),
        "malicious": int(counts.get("HIGH RISK", 0)),
        "latest": df.iloc[0].to_dict() if len(df) else None,
    }
