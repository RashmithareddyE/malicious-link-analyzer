"""
report_generator.py - Build a Markdown security report for one analysis.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import config
from modules.history_manager import sanitize_url_for_storage


def build_report(assessment, rag_explanation: Optional[Dict] = None) -> str:
    """Return the report text. Retrieved knowledge is kept apart from URL evidence."""
    a = assessment
    lines = [
        f"# {config.APP_TITLE} - Security Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**Analyzed URL (sanitised):** `{sanitize_url_for_storage(a.url)}`",
        f"**Final risk:** {a.category} ({a.final_score}/100)",
        f"**Rule-based score:** {a.rule_score}/100",
        f"**ML prediction:** {a.ml_label or 'unavailable'}"
        + (f" (ML score {a.ml_score}/100)" if a.ml_score is not None else ""),
        "",
        "## Evidence found in this URL (rule engine)",
    ]
    lines += [f"- **{i['name']}** (+{i['points']}): {i['detail']}" for i in a.indicators] \
        or ["- No suspicious indicators were triggered."]
    lines += ["", "## Score breakdown"] + [f"- {s}" for s in a.score_breakdown]
    if rag_explanation:
        lines += ["", "## Background knowledge (retrieved, NOT evidence about this URL)"]
        for k in rag_explanation.get("retrieved_knowledge", []):
            lines.append(f"- *{k['indicator']}* - {k['passage']} (source: {k['source']})")
        lines.append(f"\n_Generation mode: {rag_explanation['info']['generation_mode']}._")
    lines += ["", "## Recommended actions"] + [f"- {r}" for r in a.recommendations]
    lines += ["", "## Important", a.disclaimer, "",
              "_The URL was never opened, visited or downloaded during analysis._"]
    return "\n".join(lines)


def save_report(text: str, directory: Path = config.REPORTS_DIR) -> Path:
    """Write the report to the reports/ folder and return its path."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    path.write_text(text, encoding="utf-8")
    return path
