"""Experimental component-failure risk model trained on SYNTHETIC data.

Pipeline
--------
1. ``generate_training_data`` creates a reproducible synthetic population. A hidden
   "degradation" variable drives noisy sensor readings and, together with usage
   factors, the probability of a (synthetic) unscheduled removal within the next
   50 flying hours. The model never sees the hidden variable, so it cannot be
   perfect - the same way real sensors only partially reveal component condition.
2. A histogram gradient-boosting classifier with monotonic constraints (risk can
   only rise as any input worsens) is fitted. The data is split 75/25 (stratified) *before* fitting. Hyper-parameters are fixed
   in code, the test split is used only once for evaluation, so reported metrics are
   held-out metrics, not training accuracy.
3. Explanations: global permutation importance on the test split, and for an
   individual prediction a one-at-a-time ablation - each feature is reset to the
   median of healthy training components and the change in predicted risk is
   reported as that feature's contribution.
4. If held-out ROC-AUC falls below ``config.MIN_ACCEPTABLE_AUC`` the engine switches
   to a transparent, weighted rule-based score.

None of this is validated on real aircraft data.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import train_test_split

from . import config

FEATURES = config.FEATURES
ALERT_THRESHOLD = 0.45  # probability at which a component is flagged High or worse

# Transparent rule-based score: (nominal, limit, weight). Each feature contributes
# weight * clip((x - nominal) / (limit - nominal), 0, 1). Weights sum to 1.
RULE_SPEC = {
    "life_used_pct": (50.0, 110.0, 0.15),
    "hours_since_maintenance": (200.0, 1000.0, 0.10),
    "age_years": (5.0, 30.0, 0.05),
    "temp_deviation": (3.0, 30.0, 0.20),
    "vibration": (2.5, 10.0, 0.25),
    "pressure_deviation": (2.0, 18.0, 0.15),
    "prior_faults": (0.0, 5.0, 0.10),
}


# --------------------------------------------------------------------------- synthetic data
def simulate_sensors(degradation: np.ndarray, rng: np.random.Generator):
    """Noisy sensor channels as a function of hidden degradation (0 healthy .. 1 failed)."""
    d = np.asarray(degradation, dtype=float)
    temp = rng.normal(1.0 + 32.0 * d ** 1.4, 2.5)
    vib = np.clip(rng.normal(1.6 + 8.5 * d ** 1.2, 0.7), 0.2, None)
    pres = np.abs(rng.normal(1.0 + 17.0 * d ** 1.5, 1.8))
    return temp, vib, pres


def generate_training_data(n: int = 8000, seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    degradation = rng.beta(1.4, 4.5, n)
    # 15% drawn uniformly so the extremes reachable in the UI are represented in training
    tail = rng.random(n) < 0.15
    degradation[tail] = rng.uniform(0, 1, tail.sum())
    life = np.clip(rng.normal(55, 28, n), 0, 125)
    hsm = np.clip(rng.gamma(2.2, 150, n), 0, 1500)
    age = rng.uniform(0.5, 30, n)
    temp, vib, pres = simulate_sensors(degradation, rng)
    faults = rng.poisson(0.3 + 2.5 * degradation + age / 30)

    logit = (
        -5.2
        + 6.0 * degradation
        + 0.035 * np.maximum(life - 75, 0)
        + 0.0016 * hsm
        + 0.025 * age
        + 0.22 * faults
        + 0.9 * ((vib > 8) & (temp > 15))
        + rng.normal(0, 0.5, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    label = (rng.random(n) < prob).astype(int)
    return pd.DataFrame({
        "life_used_pct": life.round(1),
        "hours_since_maintenance": hsm.round(0),
        "age_years": age.round(1),
        "temp_deviation": temp.round(1),
        "vibration": vib.round(2),
        "pressure_deviation": pres.round(1),
        "prior_faults": faults,
        "failure_within_50h": label,
    })


# --------------------------------------------------------------------------- rule-based
def rule_contributions(X: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for feat, (nominal, limit, weight) in RULE_SPEC.items():
        out[feat] = weight * np.clip((X[feat].astype(float) - nominal) / (limit - nominal), 0, 1)
    return pd.DataFrame(out, index=X.index)


def rule_score(X: pd.DataFrame) -> np.ndarray:
    return rule_contributions(X).sum(axis=1).to_numpy()


def risk_category(score: float) -> str:
    for threshold, label in config.RISK_BANDS:
        if score >= threshold:
            return label
    return "Low"


# --------------------------------------------------------------------------- engine
@dataclass
class RiskEngine:
    model: HistGradientBoostingClassifier | None
    method: str  # "ml" or "rules"
    metrics: dict = field(default_factory=dict)
    importances: pd.DataFrame | None = None
    reference: dict = field(default_factory=dict)
    trained_at: str = ""
    n_train: int = 0
    n_test: int = 0
    version: int = config.MODEL_VERSION
    sklearn_version: str = sklearn.__version__

    @property
    def method_label(self) -> str:
        if self.method == "ml":
            return "Monotonic gradient-boosting classifier (experimental, synthetic training data)"
        return "Transparent rule-based score (fallback)"

    def _frame(self, X) -> pd.DataFrame:
        if isinstance(X, dict):
            X = pd.DataFrame([X])
        X = X[FEATURES].astype(float)
        if X.isna().any().any():
            X = X.fillna(pd.Series(self.reference))
        return X

    def predict(self, X) -> np.ndarray:
        X = self._frame(X)
        if self.method == "ml" and self.model is not None:
            return self.model.predict_proba(X)[:, 1]
        return rule_score(X)

    def predict_one(self, features: dict) -> float:
        return float(self.predict(features)[0])

    def contributions(self, X) -> pd.DataFrame:
        """Per-feature contribution for every row (positive = raises risk).

        ML: one-at-a-time ablation against the healthy reference profile.
        Rules: the weighted rule terms themselves.
        """
        X = self._frame(X).reset_index(drop=True)
        if self.method != "ml" or self.model is None:
            return rule_contributions(X)[FEATURES]
        base = self.model.predict_proba(X)[:, 1]
        blocks = []
        for feat in FEATURES:
            v = X.copy()
            v[feat] = self.reference[feat]
            blocks.append(v)
        ablated = self.model.predict_proba(pd.concat(blocks, ignore_index=True))[:, 1]
        ablated = ablated.reshape(len(FEATURES), len(X)).T
        return pd.DataFrame(base[:, None] - ablated, columns=FEATURES)

    def explain(self, features: dict) -> pd.DataFrame:
        """Per-feature contribution to one prediction, sorted by impact."""
        X = self._frame(features)
        contrib = self.contributions(X).iloc[0][FEATURES].to_numpy()
        df = pd.DataFrame({
            "feature": FEATURES,
            "label": [config.FEATURE_LABELS[f] for f in FEATURES],
            "value": X.iloc[0][FEATURES].to_numpy(),
            "reference": [self.reference[f] for f in FEATURES],
            "contribution": contrib,
        })
        return df.sort_values("contribution", ascending=False).reset_index(drop=True)


def train_engine(n: int = 8000, seed: int = config.RANDOM_SEED) -> RiskEngine:
    data = generate_training_data(n, seed)
    X, y = data[FEATURES], data["failure_within_50h"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    # Monotonic constraints encode domain knowledge: a worse value of any input may never
    # lower the predicted risk.
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
        monotonic_cst=[1] * len(FEATURES), random_state=seed,
    )
    model.fit(X_train, y_train)

    p_test = model.predict_proba(X_test)[:, 1]
    p_train = model.predict_proba(X_train)[:, 1]
    yhat = (p_test >= ALERT_THRESHOLD).astype(int)
    fpr, tpr, _ = roc_curve(y_test, p_test)
    rule_auc = roc_auc_score(y_test, rule_score(X_test))

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, p_test)),
        "average_precision": float(average_precision_score(y_test, p_test)),
        "brier": float(brier_score_loss(y_test, p_test)),
        "accuracy": float(accuracy_score(y_test, yhat)),
        "precision": float(precision_score(y_test, yhat, zero_division=0)),
        "recall": float(recall_score(y_test, yhat, zero_division=0)),
        "f1": float(f1_score(y_test, yhat, zero_division=0)),
        "confusion": confusion_matrix(y_test, yhat).tolist(),
        "threshold": ALERT_THRESHOLD,
        "prevalence_test": float(y_test.mean()),
        "majority_baseline_accuracy": float(max(y_test.mean(), 1 - y_test.mean())),
        "rule_based_roc_auc": float(rule_auc),
        "train_roc_auc": float(roc_auc_score(y_train, p_train)),
        "by_threshold": [
            {
                "threshold": t,
                "flags_at_or_above": label,
                "flagged_pct": float((p_test >= t).mean()),
                "precision": float(precision_score(y_test, p_test >= t, zero_division=0)),
                "recall": float(recall_score(y_test, p_test >= t, zero_division=0)),
                "f1": float(f1_score(y_test, p_test >= t, zero_division=0)),
            }
            for t, label in sorted(config.RISK_BANDS)[1:]
        ],
        "roc_fpr": fpr[:: max(1, len(fpr) // 200)].tolist(),
        "roc_tpr": tpr[:: max(1, len(tpr) // 200)].tolist(),
    }

    perm = permutation_importance(
        model, X_test, y_test, scoring="roc_auc", n_repeats=5, random_state=seed
    )
    importances = pd.DataFrame({
        "feature": FEATURES,
        "label": [config.FEATURE_LABELS[f] for f in FEATURES],
        "importance": perm.importances_mean,
        "std": perm.importances_std,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    reference = X_train[y_train == 0].median().to_dict()

    method = "ml" if metrics["roc_auc"] >= config.MIN_ACCEPTABLE_AUC else "rules"
    return RiskEngine(
        model=model,
        method=method,
        metrics=metrics,
        importances=importances,
        reference=reference,
        trained_at=datetime.now().replace(microsecond=0).isoformat(sep=" "),
        n_train=len(X_train),
        n_test=len(X_test),
    )


def load_or_train(path=config.MODEL_PATH, force: bool = False) -> RiskEngine:
    if not force and path.exists():
        try:
            with open(path, "rb") as fh:
                engine = pickle.load(fh)
            if (
                isinstance(engine, RiskEngine)
                and engine.version == config.MODEL_VERSION
                and engine.sklearn_version == sklearn.__version__
            ):
                return engine
        except Exception:
            pass  # corrupt or incompatible pickle -> retrain below
    engine = train_engine()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(engine, fh)
    return engine
