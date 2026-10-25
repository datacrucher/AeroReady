"""Module A - Fleet Command Centre."""
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aeroready import availability, config, db, inventory, ui
from aeroready.workorders import due_date

ui.page_header("Fleet Command Centre",
               "Live readiness of the fictional demonstration fleet - every figure is computed from the "
               "application database.")
ui.show_flash()

aircraft = db.table("aircraft")
wos = db.table("work_orders")
scored = ui.scored_components()
status = availability.aircraft_status(aircraft, wos)
k = availability.fleet_kpis(status)
stock = inventory.stock_table(scored)

active = wos[wos["status"].isin(config.ACTIVE_WO_STATUSES)].copy()
addressed = set(active["component_id"])
attention = scored[scored["risk_level"].isin(["High", "Critical"])]
unaddressed = attention[~attention["component_id"].isin(addressed)]
high_prio = active[active["severity"].isin(["High", "Critical"])]
today = date.today()
active["due"] = [due_date(s, c) for s, c in zip(active["severity"], active["created_at"])]
overdue = active[(active["due"] < today) & (active["status"] != "In Progress")]
shortages = stock[(stock["status"] == "Out of Stock") & ((stock["waiting_demand"] + stock["predicted_demand"]) > 0)]

ui.kpi_row([
    ui.kpi("Fleet availability", f"{k['availability_pct']}%", f"{k['serviceable']} of {k['total']} serviceable"),
    ui.kpi("Serviceable", k["serviceable"], "ready for tasking", "good"),
    ui.kpi("Under maintenance", k["under_maintenance"], "work in progress", "warn"),
    ui.kpi("Unavailable", k["unavailable"], "grounded, awaiting work/parts", "bad"),
    ui.kpi("Aircraft in demo fleet", k["total"], f"{len(scored)} monitored components"),
])
st.write("")
ui.kpi_row([
    ui.kpi("Components needing attention", len(attention), f"{len(unaddressed)} without a work order",
           "serious" if len(attention) else "good"),
    ui.kpi("High-priority work orders", len(high_prio), "active High / Critical", "serious" if len(high_prio) else "good"),
    ui.kpi("Awaiting parts", int((active["status"] == "Awaiting Parts").sum()), "work orders blocked by stock", "warn"),
    ui.kpi("Critical spares shortages", len(shortages), "out of stock with demand", "bad" if len(shortages) else "good"),
    ui.kpi("Overdue work orders", len(overdue), "past severity due date", "bad" if len(overdue) else "good"),
])

st.write("")
left, right = st.columns([3, 2], gap="medium")

with left:
    st.subheader("Aircraft readiness board")
    worst = scored.sort_values("risk_score", ascending=False).groupby("tail_no").first()
    board = status.merge(worst[["component_type", "risk_score", "risk_level"]], left_on="tail_no",
                         right_index=True, how="left")
    order = {"Unavailable": 0, "Under Maintenance": 1, "Serviceable": 2}
    board = board.sort_values(["status", "risk_score"], key=lambda s: s.map(order) if s.name == "status" else -s)
    st.dataframe(
        board[["tail_no", "aircraft_type", "base", "status", "risk_level", "risk_score", "component_type",
               "active_work_orders", "status_reason"]],
        hide_index=True, height=360,
        column_config={
            "tail_no": "Tail no.", "aircraft_type": "Type", "base": "Base", "status": "Status",
            "risk_level": "Worst risk", "component_type": "Worst component",
            "risk_score": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=100, format="%.0f"),
            "active_work_orders": "Active WOs", "status_reason": "Reason",
        })
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    pick = c1.selectbox("Inspect an aircraft's component health", aircraft["tail_no"], key="dash_pick")
    if c2.button("Open digital twin", type="primary", width="stretch"):
        st.session_state["sel_tail"] = pick
        st.session_state.pop("sel_comp", None)
        st.switch_page("views/digital_twin.py")

with right:
    st.subheader("Alerts")
    alerts = []
    for r in unaddressed.sort_values("risk_score", ascending=False).itertuples():
        alerts.append((r.risk_level, f"{r.component_id} {r.component_type} - risk {r.risk_score:.0f}",
                       f"No work order. {r.recommendation}"))
    for r in active[active["status"] == "Awaiting Parts"].itertuples():
        alerts.append(("Awaiting Parts", f"{r.wo_id} on {r.tail_no} blocked - {r.part_number} unavailable",
                       f"{r.severity} {r.action.lower()}" + (" · aircraft grounded" if r.grounding else "")))
    for r in shortages.itertuples():
        alerts.append(("Out of Stock", f"{r.part_number} {r.description}",
                       f"{r.waiting_demand} waiting, {r.predicted_demand} predicted demand, "
                       f"{r.on_order} on order · lead time {r.lead_time_days} d"))
    for r in overdue.itertuples():
        alerts.append(("High", f"{r.wo_id} overdue (due {r.due})", f"{r.tail_no} · status {r.status}"))
    if alerts:
        with st.container(height=420, border=False):
            st.markdown("".join(ui.alert(*a) for a in alerts), unsafe_allow_html=True)
    else:
        ui.empty_state("No active alerts.")

st.write("")
st.subheader("Health and maintenance trends")
t1, t2, t3 = st.columns(3, gap="medium")

hist = ui.history_with_risk()
with t1:
    if hist.empty:
        ui.empty_state("No sensor history available.")
    else:
        hist["day"] = hist["recorded_at"].dt.date
        daily = (hist.sort_values("recorded_at").groupby(["component_id", "day"]).last().reset_index()
                 .groupby("day").agg(health=("health_index", "mean"),
                                     at_risk=("risk_score", lambda s: int((s >= 45).sum()))).reset_index())
        fig = go.Figure(go.Scatter(x=daily["day"], y=daily["at_risk"], mode="lines", line=dict(width=2, color=ui.SERIES[0]),
                                   hovertemplate="%{x|%d %b}: %{y} components<extra></extra>"))
        fig.update_layout(title="Components at High/Critical risk (daily)", height=260, yaxis_rangemode="tozero")
        st.plotly_chart(fig, config={"displayModeBar": False})
        st.caption(f"Fleet mean health index today: **{daily['health'].iloc[-1]:.1f}** / 100 "
                   f"(30 days ago {daily['health'].iloc[0]:.1f}).")

with t2:
    rec = db.query_df("SELECT performed_on, record_type FROM maintenance_records")
    rec["performed_on"] = pd.to_datetime(rec["performed_on"], errors="coerce")
    rec = rec[rec["performed_on"] >= pd.Timestamp(today - timedelta(weeks=12))]
    if rec.empty:
        ui.empty_state("No maintenance in the last 12 weeks.")
    else:
        rec["week"] = rec["performed_on"].dt.to_period("W").dt.start_time
        wk = rec.groupby(["week", "record_type"]).size().unstack(fill_value=0)
        fig = go.Figure()
        for i, col in enumerate(["Scheduled", "Inspection", "Unscheduled"]):
            if col in wk:
                fig.add_bar(x=wk.index, y=wk[col], name=col, marker_color=ui.SERIES[i],
                            hovertemplate=f"{col}: %{{y}}<extra></extra>")
        fig.update_layout(barmode="stack", title="Maintenance actions per week (12 weeks)", height=260, bargap=0.25)
        st.plotly_chart(fig, config={"displayModeBar": False})

with t3:
    dist = scored.groupby(["component_type", "risk_level"]).size().unstack(fill_value=0)
    fig = go.Figure()
    for lvl in config.RISK_LEVELS:
        if lvl in dist:
            fig.add_bar(y=dist.index, x=dist[lvl], name=lvl, orientation="h", marker_color=ui.STATUS_COLORS[lvl],
                        hovertemplate=f"{lvl}: %{{x}}<extra></extra>")
    fig.update_layout(barmode="stack", title="Component risk by type", height=260, bargap=0.3)
    st.plotly_chart(fig, config={"displayModeBar": False})

st.subheader("Recent maintenance activity")
log = db.query_df("SELECT ts, category, tail_no, message FROM activity_log ORDER BY log_id DESC LIMIT 12")
st.dataframe(log, hide_index=True, column_config={"ts": "Time", "category": "Category", "tail_no": "Aircraft",
                                                  "message": "Activity"})
