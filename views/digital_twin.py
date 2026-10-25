"""Module D - Component Digital Twin (software simulation, not a physics model)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from aeroready import config, db, risk, ui, workorders

engine = ui.get_engine()
ui.page_header("Component Digital Twin",
               "Interactive software simulation of a fictional component: simulated telemetry, health "
               "history and degradation projection. Not a physics-validated engineering model.")
ui.show_flash()

aircraft = db.table("aircraft")
scored = ui.scored_components()
tails = aircraft["tail_no"].tolist()

c1, c2 = st.columns([1, 2])
tail = c1.selectbox("Aircraft", tails, index=tails.index(st.session_state["sel_tail"])
                    if st.session_state.get("sel_tail") in tails else 0)
st.session_state["sel_tail"] = tail
ac_comps = scored[scored["tail_no"] == tail].sort_values("risk_score", ascending=False)
ids = ac_comps["component_id"].tolist()
labels = {r["component_id"]: f"{ui.component_label(r)}  ·  risk {r['risk_score']:.0f} ({r['risk_level']})"
          for r in ac_comps.to_dict("records")}
cid = c2.selectbox("Component (highest risk first)", ids,
                   index=ids.index(st.session_state["sel_comp"]) if st.session_state.get("sel_comp") in ids else 0,
                   format_func=labels.get)
st.session_state["sel_comp"] = cid
comp = ac_comps.set_index("component_id").loc[cid].to_dict() | {"component_id": cid}

# aircraft overview strip
ac = aircraft.set_index("tail_no").loc[tail]
st.markdown(
    f"<div class='ar-panel' style='margin:.4rem 0 .8rem 0'><b>{tail}</b> · {ac['aircraft_type']} · {ac['base']} · "
    f"{ac['squadron']} · built {ac['year_built']} · {ac['total_flight_hours']:,.0f} FH<br>"
    + " ".join(f"<span title='{r.component_type} {r.position}' style='margin-right:.35rem'>"
               f"<span class='ar-tag'>{r.component_type} {r.position}</span> {ui.badge(r.risk_level)}</span>"
               for r in ac_comps.itertuples()) + "</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------- identity + controls
left, mid, right = st.columns([1.2, 1.4, 1.6], gap="medium")
with left:
    st.markdown("##### Identity & usage")
    ident = pd.DataFrame({
        "Field": ["Component ID", "Type / position", "System", "Serial no.", "Installed", "Operating hours",
                  "Rated life", "Life consumed", "Since last maintenance", "Maintenance events", "Prior faults (24 mo)",
                  "Component age", "Last reading"],
        "Value": [cid, f"{comp['component_type']} {comp['position']}",
                  config.COMPONENT_TYPES[comp["component_type"]]["system"], comp["serial_no"], comp["install_date"],
                  f"{comp['operating_hours']:,.0f} h", f"{comp['rated_life_hours']:,.0f} h",
                  f"{comp['life_used_pct']:.0f}%", f"{comp['hours_since_maintenance']:,.0f} h",
                  str(comp["maintenance_count"]), str(comp["prior_faults"]), f"{comp['age_years']:.1f} years",
                  str(comp["last_reading_at"])],
    })
    st.dataframe(ident, hide_index=True, height=490)

with mid:
    st.markdown("##### Simulation controls")
    st.caption("Recorded values are pre-loaded. Move a control to simulate a new condition; risk is recomputed "
               "immediately.")
    sim = {}
    for f in config.SENSOR_FEATURES:
        lo, hi, step = config.FEATURE_RANGES[f]
        sim[f] = st.slider(config.FEATURE_LABELS[f], lo, hi, float(min(max(comp[f], lo), hi)), step, key=f"tw_{cid}_{f}")
    extra = st.slider("Simulate additional flying hours", 0, 500, 0, 10, key=f"tw_{cid}_extra",
                      help="Adds usage to operating hours and hours since last maintenance.")
    noise = st.toggle("Stream live-style telemetry (simulated sensor noise)", key=f"tw_{cid}_live")

features = {
    "life_used_pct": (comp["operating_hours"] + extra) / comp["rated_life_hours"] * 100,
    "hours_since_maintenance": comp["hours_since_maintenance"] + extra,
    "age_years": comp["age_years"],
    "prior_faults": comp["prior_faults"],
    **sim,
}
recorded = {f: comp[f] for f in config.FEATURES}
current = risk.assess(engine, features)

with right:
    @st.fragment(run_every=2 if noise else None)
    def live_panel():
        live = dict(features)
        if noise:
            rng = np.random.default_rng()
            live["temp_deviation"] += rng.normal(0, 0.8)
            live["vibration"] = max(0.0, live["vibration"] + rng.normal(0, 0.25))
            live["pressure_deviation"] = abs(live["pressure_deviation"] + rng.normal(0, 0.5))
            buf = st.session_state.setdefault(f"buf_{cid}", [])
            buf.append({f: live[f] for f in config.SENSOR_FEATURES})
            del buf[:-40]
        a = risk.assess(engine, live)
        st.markdown("##### Current assessment")
        ui.assessment_card(a, engine.method_label)
        base = risk.assess(engine, recorded)
        if abs(a["score"] - base["score"]) >= 0.1 and not noise:
            st.caption(f"Recorded condition: {base['score']:.0f} ({base['level']}). Simulated change "
                       f"{a['score'] - base['score']:+.1f} points.")
        m1, m2, m3 = st.columns(3)
        m1.metric("Temp. dev.", f"{live['temp_deviation']:.1f} °C", f"{live['temp_deviation'] - comp['temp_deviation']:+.1f}",
                  delta_color="inverse")
        m2.metric("Vibration", f"{live['vibration']:.2f} mm/s", f"{live['vibration'] - comp['vibration']:+.2f}",
                  delta_color="inverse")
        m3.metric("Pressure dev.", f"{live['pressure_deviation']:.1f} %",
                  f"{live['pressure_deviation'] - comp['pressure_deviation']:+.1f}", delta_color="inverse")
        if noise and st.session_state.get(f"buf_{cid}"):
            b = pd.DataFrame(st.session_state[f"buf_{cid}"])
            fig = go.Figure(go.Scatter(y=b["vibration"], mode="lines", line=dict(width=2, color=ui.SERIES[0]),
                                       hovertemplate="%{y:.2f} mm/s<extra></extra>"))
            fig.update_layout(title="Live vibration stream (simulated, last 40 samples)", height=170,
                              xaxis_visible=False)
            st.plotly_chart(fig, config={"displayModeBar": False})
        st.plotly_chart(ui.explanation_chart(a["explanation"], height=250), config={"displayModeBar": False})

    live_panel()

b1, b2, _ = st.columns([1.3, 1.3, 2])
if b1.button("Commit simulated reading to record", type="primary", width="stretch",
             help="Saves the three sensor values as a new reading; risk across the app updates."):
    try:
        workorders.record_reading(cid, sim["temp_deviation"], sim["vibration"], sim["pressure_deviation"])
    except workorders.WorkOrderError as exc:
        st.error(str(exc))
    else:
        for f in config.SENSOR_FEATURES:
            st.session_state.pop(f"tw_{cid}_{f}", None)
        ui.flash("success", f"Reading committed for {cid}. Fleet risk, alerts and planner now reflect it.")
        st.rerun()
if b2.button("Reset controls to recorded values", width="stretch"):
    for key in [k for k in st.session_state if k.startswith(f"tw_{cid}_")]:
        st.session_state.pop(key)
    st.rerun()

with st.expander("Create maintenance work order for this component",
                 expanded=current["level"] in ("High", "Critical")):
    ui.work_order_panel(comp, current, key=f"tw_{cid}")

# --------------------------------------------------------------------------- history & projection
st.subheader("Health history and degradation projection")
hist = ui.history_with_risk([cid])
if hist.empty:
    ui.empty_state("No sensor history recorded for this component.")
else:
    horizon = st.slider("Projection horizon (days)", 0, 60, 30, 5)
    # Linear trend of the last 14 days of each channel, extrapolated - an illustrative assumption only.
    recent = hist[hist["recorded_at"] >= hist["recorded_at"].max() - pd.Timedelta(days=14)]
    t = (recent["recorded_at"] - recent["recorded_at"].min()).dt.total_seconds() / 86400
    slopes = {}
    for f in config.SENSOR_FEATURES:
        slopes[f] = float(np.polyfit(t, recent[f], 1)[0]) if len(recent) >= 3 and t.nunique() > 1 else 0.0
    span = (hist["recorded_at"].max() - hist["recorded_at"].min()).days or 1
    util = max((hist["operating_hours"].iloc[-1] - hist["operating_hours"].iloc[0]) / span, 0.5) \
        if hist["operating_hours"].iloc[-1] > hist["operating_hours"].iloc[0] else 1.5
    proj_rows = []
    last_day = hist["recorded_at"].max()
    for d in range(0, horizon + 1):
        f = dict(features)
        for ch in config.SENSOR_FEATURES:
            lo, hi, _ = config.FEATURE_RANGES[ch]
            f[ch] = float(np.clip(sim[ch] + slopes[ch] * d, lo, hi))
        f["life_used_pct"] = (comp["operating_hours"] + extra + util * d) / comp["rated_life_hours"] * 100
        f["hours_since_maintenance"] = comp["hours_since_maintenance"] + extra + util * d
        proj_rows.append({"date": last_day + pd.Timedelta(days=d), **f})
    proj = pd.DataFrame(proj_rows)
    proj["risk_score"] = engine.predict(proj[config.FEATURES]) * 100
    proj["health_index"] = 100 - proj["risk_score"]

    fig = go.Figure()
    fig.add_scatter(x=hist["recorded_at"], y=hist["health_index"], mode="lines+markers", name="Recorded",
                    line=dict(width=2, color=ui.SERIES[0]), marker=dict(size=4),
                    hovertemplate="%{x|%d %b %H:%M}: health %{y:.0f}<extra></extra>")
    if horizon:
        fig.add_scatter(x=proj["date"], y=proj["health_index"], mode="lines", name="Projected (linear trend)",
                        line=dict(width=2, dash="dash", color=ui.SERIES[1]),
                        hovertemplate="%{x|%d %b}: health %{y:.0f}<extra></extra>")
    for thr, lvl in [(55, "High"), (30, "Critical")]:
        fig.add_hline(y=thr, line=dict(color=ui.STATUS_COLORS[lvl], width=1, dash="dot"),
                      annotation_text=f"{lvl} risk below {thr}", annotation_font_color=ui.MUTED,
                      annotation_position="bottom left")
    fig.update_layout(title="Health index (100 - risk score)", height=320, yaxis_range=[0, 102])
    st.plotly_chart(fig, config={"displayModeBar": False})
    cross = proj[proj["risk_score"] >= 45]
    if horizon and not cross.empty:
        days = (cross["date"].iloc[0] - last_day).days
        st.warning("The simulated condition is already at **High** risk or above." if days <= 0 else
                   f"On the current simulated trend the component reaches **High** risk in about **{days} day(s)** "
                   f"(assuming ~{util:.1f} FH/day).")
    elif horizon:
        st.success(f"No High-risk crossing projected within {horizon} days on the current trend.")
    st.caption("Projection = simulated sensor values extrapolated with the linear trend of the last 14 days of "
               "readings, plus estimated daily utilisation. Illustrative only.")

    fig = make_subplots(rows=1, cols=3, subplot_titles=[config.FEATURE_LABELS[f] for f in config.SENSOR_FEATURES])
    for i, f in enumerate(config.SENSOR_FEATURES, start=1):
        fig.add_scatter(x=hist["recorded_at"], y=hist[f], mode="lines", line=dict(width=2, color=ui.SERIES[0]),
                        showlegend=False, row=1, col=i, hovertemplate="%{x|%d %b}: %{y:.2f}<extra></extra>")
    fig.update_layout(height=230, margin=dict(t=40))
    fig.update_annotations(font_size=12, font_color=ui.MUTED)
    st.plotly_chart(fig, config={"displayModeBar": False})

h1, h2 = st.columns(2, gap="medium")
with h1:
    st.markdown("##### Maintenance history")
    mr = db.query_df("SELECT performed_on, record_type, description, team_id, work_order_id FROM maintenance_records "
                     "WHERE component_id = ? ORDER BY performed_on DESC", (cid,))
    if mr.empty:
        ui.empty_state("No maintenance records.")
    else:
        st.dataframe(mr, hide_index=True)
with h2:
    st.markdown("##### Faults and work orders")
    fl = db.query_df("SELECT fault_id, reported_on, fault_code, severity, status FROM faults WHERE component_id = ? "
                     "ORDER BY reported_on DESC", (cid,))
    if fl.empty:
        ui.empty_state("No fault records.")
    else:
        st.dataframe(fl, hide_index=True)
    wo = db.query_df("SELECT wo_id, action, severity, status, scheduled_start FROM work_orders WHERE component_id = ? "
                     "ORDER BY created_at DESC", (cid,))
    if not wo.empty:
        st.dataframe(wo, hide_index=True)
