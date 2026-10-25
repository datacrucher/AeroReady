"""Fleet-level risk scoring and maintenance recommendations.

Turns component records into model features, scores them with the RiskEngine and
attaches a category, the main contributing factors and a recommended action.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .model import RiskEngine, risk_category

FACTOR_ADVICE = {
    "vibration": "elevated vibration - inspect bearings, mounts and rotor balance",
    "temp_deviation": "temperature deviation - check cooling, fluid condition and sensor calibration",
    "pressure_deviation": "pressure deviation - leak-check seals, lines and filters",
    "life_used_pct": "high share of rated life consumed - plan replacement",
    "hours_since_maintenance": "long interval since last maintenance - perform overdue servicing",
    "prior_faults": "repeat-fault history - review fault records for a recurring defect",
    "age_years": "component age - include in condition inspection",
}

# category -> (default action, due within days, grounds aircraft)
CATEGORY_POLICY = {
    "Critical": ("Replace", 1, True),
    "High": ("Inspect & Service", 3, False),
    "Medium": ("Inspect & Service", 14, False),
    "Low": (None, None, False),
}

MIN_FACTOR_CONTRIBUTION = 0.02


def to_features(components: pd.DataFrame) -> pd.DataFrame:
    """Derive model features from component records (handles missing values)."""
    df = pd.DataFrame(index=components.index)
    rated = components["rated_life_hours"].replace(0, np.nan)
    df["life_used_pct"] = (components["operating_hours"] / rated * 100).round(1)
    for col in ["hours_since_maintenance", "age_years", "temp_deviation", "vibration",
                "pressure_deviation", "prior_faults"]:
        df[col] = pd.to_numeric(components[col], errors="coerce")
    return df[config.FEATURES]


def top_factors(contrib_row: pd.Series, k: int = 3) -> list[str]:
    s = contrib_row.sort_values(ascending=False)
    return [f for f, v in s.items() if v >= MIN_FACTOR_CONTRIBUTION][:k]


def recommend(level: str, factors: list[str], features: dict | pd.Series) -> dict:
    action, due_days, grounding = CATEGORY_POLICY[level]
    if action == "Inspect & Service" and float(features.get("life_used_pct", 0) or 0) >= 100:
        action = "Replace"  # beyond rated life: servicing is not enough
    if level == "Low":
        text = "No action required - continue routine monitoring."
    else:
        reasons = "; ".join(FACTOR_ADVICE[f] for f in factors) or "combined effect of several moderate indicators"
        lead = {
            "Critical": "Ground aircraft and replace component within 24 h.",
            "High": "Schedule inspection within 72 h.",
            "Medium": "Inspect at next scheduled maintenance opportunity (within 14 days).",
        }[level]
        if action == "Replace" and level != "Critical":
            lead += " Component is beyond rated life - plan replacement."
        text = f"{lead} Main drivers: {reasons}."
    return {"action": action, "due_days": due_days, "grounding": grounding, "text": text}


def score_components(engine: RiskEngine, components: pd.DataFrame) -> pd.DataFrame:
    """Return the components frame with risk columns added."""
    if components.empty:
        return components.assign(risk_prob=[], risk_score=[], risk_level=[], health_index=[],
                                 top_factors=[], recommended_action=[], recommendation=[],
                                 due_days=[], grounding=[], life_used_pct=[])
    comps = components.reset_index(drop=True)
    X = to_features(comps)
    probs = engine.predict(X)
    contrib = engine.contributions(X)
    out = comps.copy()
    out["life_used_pct"] = X["life_used_pct"]
    out["risk_prob"] = probs
    out["risk_score"] = (probs * 100).round(1)
    out["risk_level"] = [risk_category(p) for p in probs]
    out["health_index"] = (100 - out["risk_score"]).round(1)
    factors, actions, texts, dues, grounds = [], [], [], [], []
    for i in range(len(out)):
        f = top_factors(contrib.iloc[i])
        rec = recommend(out.at[i, "risk_level"], f, X.iloc[i])
        factors.append(", ".join(config.FEATURE_LABELS[x] for x in f))
        actions.append(rec["action"])
        texts.append(rec["text"])
        dues.append(rec["due_days"])
        grounds.append(rec["grounding"])
    out["top_factors"] = factors
    out["recommended_action"] = actions
    out["recommendation"] = texts
    out["due_days"] = dues
    out["grounding"] = grounds
    return out


def assess(engine: RiskEngine, features: dict) -> dict:
    """Single interactive assessment used by the what-if and digital-twin screens."""
    prob = engine.predict_one(features)
    level = risk_category(prob)
    explanation = engine.explain(features)
    factors = [r.feature for r in explanation.itertuples() if r.contribution >= MIN_FACTOR_CONTRIBUTION][:3]
    rec = recommend(level, factors, features)
    return {
        "prob": prob,
        "score": round(prob * 100, 1),
        "level": level,
        "health_index": round(100 - prob * 100, 1),
        "explanation": explanation,
        "factors": factors,
        **rec,
    }
