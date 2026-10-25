"""Module C - AI Predictive Maintenance."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aeroready import config, csv_io, db, model, risk, ui, workorders

engine = ui.get_engine()
ui.page_header("Predictive Maintenance",
               "Component-level failure-risk estimates with explanations and recommended actions. "
               "Experimental: trained and evaluated on synthetic data only.")
ui.show_flash()

if engine.method != "ml":
    st.warning("The ML model did not meet the minimum held-out ROC-AUC "
               f"({config.MIN_ACCEPTABLE_AUC}); risk is being computed with the transparent rule-based score.")

scored = ui.scored_components()
wos = db.table("work_orders")
active = wos[wos["status"].isin(config.ACTIVE_WO_STATUSES)]
scored = scored.merge(active[["component_id", "wo_id", "status"]].rename(columns={"status": "wo_status"}),
                      on="component_id", how="left")

tab_rank, tab_whatif, tab_card = st.tabs(["Fleet risk ranking", "What-if assessment", "Model card & evaluation"])

# --------------------------------------------------------------------------- ranking
with tab_rank:
    f1, f2, f3, f4 = st.columns([2, 2, 2, 1.4], vertical_alignment="bottom")
    levels = f1.multiselect("Risk level", config.RISK_LEVELS, default=["Medium", "High", "Critical"])
    types = f2.multiselect("Component type", list(config.COMPONENT_TYPES))
    tails = f3.multiselect("Aircraft", sorted(scored["tail_no"].unique()))
    only_open = f4.toggle("Only without work order", value=False)
    view = scored[scored["risk_level"].isin(levels or config.RISK_LEVELS)]
    if types:
        view = view[view["component_type"].isin(types)]
    if tails:
        view = view[view["tail_no"].isin(tails)]
    if only_open:
        view = view[view["wo_id"].isna()]
    view = view.sort_values("risk_score", ascending=False)

    counts = scored["risk_level"].value_counts()
    ui.kpi_row([ui.kpi(f"{lvl} risk", int(counts.get(lvl, 0)), "components",
                       {"Low": "good", "Medium": "warn", "High": "serious", "Critical": "bad"}[lvl])
                for lvl in config.RISK_LEVELS])
    st.write("")

    if view.empty:
        ui.empty_state("No components match these filters.")
    else:
        sel = st.dataframe(
            view[["component_id", "tail_no", "component_type", "risk_score", "risk_level", "top_factors",
                  "recommended_action", "wo_id", "wo_status"]],
            hide_index=True, on_select="rerun", selection_mode="single-row", key="rank_table", height=380,
            column_config={
                "component_id": "Component", "tail_no": "Aircraft", "component_type": "Type",
                "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
                "risk_level": "Category", "top_factors": "Main contributing factors",
                "recommended_action": "Suggested action", "wo_id": "Work order", "wo_status": "WO status",
            })
        st.caption("Select a row to see the explanation and raise a work order.")
        rows = sel.selection.rows if sel and sel.selection else []
        if rows:
            comp = view.iloc[rows[0]].to_dict()
            feats = risk.to_features(pd.DataFrame([comp])).iloc[0].to_dict()
            a = risk.assess(engine, feats)
            st.divider()
            st.markdown(f"#### {comp['component_id']} · {comp['component_type']} {comp['position']} on {comp['tail_no']}")
            c1, c2 = st.columns([2, 3], gap="medium")
            with c1:
                ui.assessment_card(a, engine.method_label)
            with c2:
                st.plotly_chart(ui.explanation_chart(a["explanation"]), config={"displayModeBar": False})
            with st.expander("Create maintenance work order", expanded=a["level"] in ("High", "Critical")):
                ui.work_order_panel(comp, a, key=f"rank_{comp['component_id']}")

    pending = scored[scored["risk_level"].isin(["High", "Critical"]) & scored["wo_id"].isna()]
    st.divider()
    b1, b2 = st.columns([3, 1], vertical_alignment="center")
    b1.markdown(f"**{len(pending)}** High/Critical component(s) have no work order. Bulk creation uses each "
                "component's recommended action and default spare part; Critical findings ground the aircraft.")
    if b2.button("Create all recommended work orders", disabled=pending.empty, width="stretch"):
        made, blocked = [], 0
        for r in pending.to_dict("records"):
            try:
                res = workorders.create_work_order(
                    r["tail_no"], r["component_id"], r["recommended_action"] or "Inspect & Service", r["risk_level"],
                    grounding=bool(r["grounding"]), risk_score=r["risk_score"],
                    recommended_action=r["recommendation"], source="Predictive")
                made.append(res["wo_id"])
                blocked += 0 if res["parts_reserved"] else 1
            except workorders.WorkOrderError as exc:
                ui.flash("error", str(exc))
        ui.flash("success", f"Created {len(made)} work order(s): {', '.join(made)}."
                 + (f" {blocked} are awaiting parts." if blocked else ""))
        st.rerun()

# --------------------------------------------------------------------------- what-if
with tab_whatif:
    st.caption("Adjust any input to recompute the risk instantly. What-if values are not saved - use the "
               "Digital Twin to commit a simulated reading to the record.")
    labels = {r["component_id"]: ui.component_label(r) for r in scored.to_dict("records")}
    ids = scored.sort_values("risk_score", ascending=False)["component_id"].tolist()
    default = st.session_state.get("sel_comp")
    cid = st.selectbox("Component", ids, index=ids.index(default) if default in ids else 0,
                       format_func=labels.get, key="whatif_comp")
    comp = scored.set_index("component_id").loc[cid].to_dict() | {"component_id": cid}
    base = risk.to_features(pd.DataFrame([comp])).iloc[0].to_dict()

    left, right = st.columns([2, 3], gap="medium")
    with left:
        feats = {}
        for f in config.FEATURES:
            lo, hi, step = config.FEATURE_RANGES[f]
            val = float(base[f]) if pd.notna(base[f]) else float(engine.reference[f])
            val = min(max(val, lo), hi)
            if f == "prior_faults":
                feats[f] = st.slider(config.FEATURE_LABELS[f], int(lo), int(hi), int(val), 1, key=f"wi_{cid}_{f}")
            else:
                feats[f] = st.slider(config.FEATURE_LABELS[f], lo, hi, round(val / step) * step, step,
                                     key=f"wi_{cid}_{f}")
    a = risk.assess(engine, feats)
    base_a = risk.assess(engine, base)
    with right:
        ui.assessment_card(a, engine.method_label)
        delta = a["score"] - base_a["score"]
        if abs(delta) >= 0.1:
            st.caption(f"Recorded condition scores {base_a['score']:.0f} ({base_a['level']}); "
                       f"what-if changes the score by {delta:+.1f} points.")
        st.plotly_chart(ui.explanation_chart(a["explanation"]), config={"displayModeBar": False})
    with st.expander("Create a work order from this assessment"):
        ui.work_order_panel(comp, a, key=f"wi_{cid}")

# --------------------------------------------------------------------------- model card
with tab_card:
    m = engine.metrics
    st.markdown(
        f"**Model:** {engine.method_label}  \n"
        f"**Target:** synthetic label 'unscheduled removal within the next 50 flying hours'.  \n"
        f"**Training data:** {engine.n_train + engine.n_test:,} reproducibly generated synthetic component records "
        f"(seed {config.RANDOM_SEED}); stratified split - {engine.n_train:,} train / {engine.n_test:,} held-out test. "
        f"Hyper-parameters fixed in code; the test split is used once for the metrics below.  \n"
        f"**Trained:** {engine.trained_at} · scikit-learn {engine.sklearn_version}")
    st.warning(
        "These metrics measure how well the model recovers the rules of the synthetic data generator. "
        "They say nothing about real-world failure prediction, which would require real labelled "
        "maintenance and health-monitoring data and independent validation.")
    ui.kpi_row([
        ui.kpi("ROC-AUC (test)", f"{m['roc_auc']:.3f}", f"train {m['train_roc_auc']:.3f}"),
        ui.kpi("Average precision (test)", f"{m['average_precision']:.3f}", f"prevalence {m['prevalence_test']:.1%}"),
        ui.kpi("Brier score (test)", f"{m['brier']:.3f}", "lower is better"),
        ui.kpi("Rule-based ROC-AUC (test)", f"{m['rule_based_roc_auc']:.3f}", "transparent baseline"),
    ])
    diff = m["roc_auc"] - m["rule_based_roc_auc"]
    st.info(f"On this synthetic test set the ML model's ROC-AUC is {abs(diff):.3f} "
            f"{'above' if diff >= 0 else 'below'} the transparent rule-based score - "
            + ("i.e. comparable performance; the rule score remains a defensible, explainable alternative."
               if abs(diff) < 0.02 else "a material difference."))
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        fig = go.Figure()
        fig.add_scatter(x=m["roc_fpr"], y=m["roc_tpr"], mode="lines", name="Model", line=dict(width=2, color=ui.SERIES[0]))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(width=1, dash="dot", color=ui.MUTED))
        fig.update_layout(title="ROC curve (held-out synthetic test set)", height=320,
                          xaxis_title="False positive rate", yaxis_title="True positive rate")
        st.plotly_chart(fig, config={"displayModeBar": False})
    with c2:
        imp = engine.importances.sort_values("importance")
        fig = go.Figure(go.Bar(x=imp["importance"], y=imp["label"], orientation="h", marker_color=ui.SERIES[0],
                               error_x=dict(type="data", array=imp["std"], color=ui.MUTED),
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(title="Permutation importance (drop in test ROC-AUC)", height=320)
        st.plotly_chart(fig, config={"displayModeBar": False})
    c3, c4 = st.columns(2, gap="medium")
    with c3:
        st.markdown("**Operating points by risk band** (held-out test set)")
        bt = pd.DataFrame(m["by_threshold"])
        st.dataframe(bt, hide_index=True, column_config={
            "threshold": st.column_config.NumberColumn("Prob. threshold", format="%.2f"),
            "flags_at_or_above": "Flags at or above",
            "flagged_pct": st.column_config.NumberColumn("Share flagged", format="percent"),
            "precision": st.column_config.NumberColumn("Precision", format="%.2f"),
            "recall": st.column_config.NumberColumn("Recall", format="%.2f"),
            "f1": st.column_config.NumberColumn("F1", format="%.2f")})
        st.caption("Lower thresholds catch more failures (recall) at the cost of more false alarms (precision).")
    with c4:
        cm = m["confusion"]
        st.markdown(f"**Confusion matrix at the High threshold ({m['threshold']})**")
        st.dataframe(pd.DataFrame(cm, index=["Actual: no failure", "Actual: failure"],
                                  columns=["Predicted: no", "Predicted: yes"]))
        st.caption(f"Accuracy {m['accuracy']:.3f} vs. {m['majority_baseline_accuracy']:.3f} for always predicting "
                   "'no failure' - accuracy alone is misleading with imbalanced classes.")
    with st.expander("Explanation method and limitations"):
        st.markdown(
            "- **Global importance:** permutation importance on the held-out test split.\n"
            "- **Per-prediction explanation:** each feature is reset to the median of healthy training "
            "components; the resulting change in predicted risk is that feature's contribution. Contributions "
            "do not sum exactly to the score because features interact.\n"
            "- **Rule-based fallback:** weighted, clipped distance of each feature from a nominal value "
            f"(weights {', '.join(f'{config.FEATURE_LABELS[k]} {v[2]:.2f}' for k, v in model.RULE_SPEC.items())}).\n"
            "- Sensor values, limits and failure mechanisms are invented; the model is not physics-based.\n"
            "- Risk categories: Low < 0.20 ≤ Medium < 0.45 ≤ High < 0.70 ≤ Critical (probability).")
        st.download_button("Download synthetic training data (CSV)",
                           csv_io.to_csv_bytes(model.generate_training_data()),
                           "aeroready_synthetic_training_data.csv", "text/csv")
