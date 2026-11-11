"""
risk_assessment.py - Combine rule-based and ML results into one final risk assessment.

Transparent scoring (all numbers live in config.SCORING):
    final = rule_weight * rule_score + ml_weight * ml_score     (0-100)
then safety floors are applied so strong evidence can never end up as LOW RISK.

The RAG component contributes ONLY explanations. It has no input into the score and
cannot declare a URL safe.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import config
from modules.url_analyzer import RuleAnalysis

DISCLAIMER = ("A LOW RISK result does not guarantee that a website is safe. This tool only "
              "inspects the text of the link; it does not visit the site, check reputation "
              "databases or inspect page content. New or well-disguised attacks can look normal.")

RECOMMENDATIONS = {
    "LOW RISK": [
        "No strong warning signs were found in the link text, but stay alert.",
        "Check that the domain exactly matches the organisation you expect.",
        "Do not enter passwords, OTPs or card details unless you reached the site on purpose.",
        "Prefer typing the address or using a bookmark for banking and account pages.",
    ],
    "MEDIUM RISK": [
        "Do not open this link on a device that holds sensitive data until it is verified.",
        "Verify the link through another channel (ask the sender, or use the official app/website).",
        "Never enter credentials, OTPs or payment details on a page reached from this link.",
        "If you must check it, use a separate, isolated environment or a trusted URL scanner.",
    ],
    "HIGH RISK": [
        "Do NOT open this link and do not enter any information on it.",
        "Delete the message / avoid the printed QR code and warn others who may have seen it.",
        "If you already visited it, change passwords, enable multi-factor authentication and contact your bank if needed.",
        "Report the link to the impersonated brand, your IT/security team or your national cyber-crime portal.",
    ],
}


@dataclass
class RiskAssessment:
    url: str
    final_score: int
    category: str
    rule_score: int
    ml_score: Optional[float]
    ml_label: Optional[str]
    ml_probabilities: Dict[str, float] = field(default_factory=dict)
    indicators: List[Dict] = field(default_factory=list)       # triggered rules
    passed_checks: List[str] = field(default_factory=list)
    score_breakdown: List[str] = field(default_factory=list)   # human-readable steps
    recommendations: List[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER
    ml_available: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def categorize(score: float, cfg: Dict = None) -> str:
    """Map a 0-100 score to LOW / MEDIUM / HIGH RISK using configured thresholds."""
    cfg = cfg or config.SCORING
    if score >= cfg["high_threshold"]:
        return "HIGH RISK"
    if score >= cfg["medium_threshold"]:
        return "MEDIUM RISK"
    return "LOW RISK"


def assess(rule_analysis: RuleAnalysis, ml_result: Optional[Dict] = None,
           cfg: Dict = None) -> RiskAssessment:
    """Combine the rule analysis and (optional) ML prediction into a final assessment."""
    cfg = cfg or config.SCORING
    rule_score = rule_analysis.rule_score
    steps: List[str] = []

    rw, mw = cfg["rule_weight"], cfg["ml_weight"]
    if ml_result is None:
        # Model unavailable: rules carry the whole weight (never silently assume 'safe')
        score = float(rule_score)
        steps.append(f"ML model unavailable -> final = rule score = {rule_score}")
    else:
        total = rw + mw
        score = (rw / total) * rule_score + (mw / total) * ml_result["ml_score"]
        steps.append(f"Weighted score = {rw/total:.2f} x rule({rule_score}) + "
                     f"{mw/total:.2f} x ML({ml_result['ml_score']}) = {score:.1f}")

    # ---- safety floors: strong evidence can only push the score UP ----
    triggered_ids = {c.rule_id for c in rule_analysis.triggered}
    critical = sorted(triggered_ids & config.CRITICAL_RULES)
    if critical and score < cfg["floor_if_critical_rule"]:
        score = cfg["floor_if_critical_rule"]
        steps.append(f"Critical rule(s) fired ({', '.join(critical)}) -> score raised to at least "
                     f"{cfg['floor_if_critical_rule']}")
    if ml_result and ml_result["label"] == "MALICIOUS" and score < cfg["floor_if_ml_malicious"]:
        score = cfg["floor_if_ml_malicious"]
        steps.append(f"ML predicted MALICIOUS -> score raised to at least {cfg['floor_if_ml_malicious']}")
    if rule_score >= cfg["rule_high_cutoff"] and score < cfg["floor_if_rule_high"]:
        score = cfg["floor_if_rule_high"]
        steps.append(f"Rule score >= {cfg['rule_high_cutoff']} -> score raised to at least "
                     f"{cfg['floor_if_rule_high']}")

    final = int(round(max(0, min(100, score))))
    category = categorize(final, cfg)
    steps.append(f"Final score {final} -> {category} "
                 f"(MEDIUM >= {cfg['medium_threshold']}, HIGH >= {cfg['high_threshold']})")
    steps.append("RAG explanations are informational only and did not change the score.")

    return RiskAssessment(
        url=rule_analysis.url,
        final_score=final,
        category=category,
        rule_score=rule_score,
        ml_score=ml_result["ml_score"] if ml_result else None,
        ml_label=ml_result["label"] if ml_result else None,
        ml_probabilities=ml_result["probabilities"] if ml_result else {},
        indicators=[{"rule_id": c.rule_id, "name": c.name, "points": c.points,
                     "severity": c.severity, "detail": c.detail} for c in rule_analysis.triggered],
        passed_checks=[c.name for c in rule_analysis.checks if not c.triggered],
        score_breakdown=steps,
        recommendations=RECOMMENDATIONS[category],
        ml_available=ml_result is not None,
    )
