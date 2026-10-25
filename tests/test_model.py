import numpy as np
import pandas as pd

from aeroready import config, model, risk


def test_training_data_is_reproducible():
    a = model.generate_training_data(500, seed=7)
    b = model.generate_training_data(500, seed=7)
    pd.testing.assert_frame_equal(a, b)
    assert set(a["failure_within_50h"].unique()) <= {0, 1}


def test_metrics_are_held_out_and_reasonable(engine):
    m = engine.metrics
    assert engine.n_test > 0 and engine.n_train > 0
    assert abs(engine.n_test / (engine.n_train + engine.n_test) - 0.25) < 0.01
    assert 0.5 < m["roc_auc"] <= 1.0
    # held-out metric is reported separately from training performance
    assert m["train_roc_auc"] >= m["roc_auc"] - 0.05
    assert engine.method == ("ml" if m["roc_auc"] >= config.MIN_ACCEPTABLE_AUC else "rules")


def test_risk_is_monotonic_in_every_feature(engine):
    base = dict(life_used_pct=50, hours_since_maintenance=250, age_years=10, prior_faults=1,
                temp_deviation=3, vibration=2.5, pressure_deviation=2)
    for feat in config.FEATURES:
        lo, hi, _ = config.FEATURE_RANGES[feat]
        grid = [dict(base, **{feat: v}) for v in np.linspace(lo, hi, 15)]
        p = engine.predict(pd.DataFrame(grid))
        assert np.all(np.diff(p) >= -1e-9), f"risk decreases as {feat} increases"


def test_risk_category_bands():
    assert model.risk_category(0.0) == "Low"
    assert model.risk_category(0.19) == "Low"
    assert model.risk_category(0.20) == "Medium"
    assert model.risk_category(0.45) == "High"
    assert model.risk_category(0.70) == "Critical"
    assert model.risk_category(1.0) == "Critical"


def test_rule_score_bounded_and_fallback(engine):
    data = model.generate_training_data(300, seed=1)[config.FEATURES]
    s = model.rule_score(data)
    assert s.min() >= 0 and s.max() <= 1
    fallback = model.RiskEngine(model=None, method="rules", reference=engine.reference)
    assert np.allclose(fallback.predict(data), s)
    expl = fallback.explain(data.iloc[0].to_dict())
    assert np.isclose(expl["contribution"].sum(), s[0])


def test_assessment_responds_to_sensor_changes(engine):
    healthy = dict(life_used_pct=40, hours_since_maintenance=150, age_years=6, prior_faults=0,
                   temp_deviation=2, vibration=2.0, pressure_deviation=1.5)
    worse = dict(healthy, temp_deviation=35, vibration=11, pressure_deviation=20)
    a, b = risk.assess(engine, healthy), risk.assess(engine, worse)
    assert b["score"] > a["score"]
    assert config.SEVERITY_RANK[b["level"]] > config.SEVERITY_RANK[a["level"]]
    assert b["factors"], "a high-risk assessment should name contributing factors"
    assert "Main drivers" in b["text"]


def test_missing_values_are_imputed(engine):
    row = dict(life_used_pct=np.nan, hours_since_maintenance=100, age_years=np.nan, prior_faults=0,
               temp_deviation=2, vibration=2, pressure_deviation=1)
    assert 0 <= engine.predict_one(row) <= 1
