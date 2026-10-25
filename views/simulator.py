"""Module G - Fleet Availability Simulator."""
from datetime import date

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from aeroready import availability, db, ui

ui.page_header("Fleet Availability Simulator",
               "Compares maintenance strategies on the fictional fleet using the same scheduling heuristic. "
               "Results depend on the stated assumptions; they are not forecasts of real-world downtime.")
ui.show_flash()

today = date.today()
scored = ui.scored_components()
aircraft = db.table("aircraft")
wos = db.table("work_orders")
status = availability.aircraft_status(aircraft, wos)
k = availability.fleet_kpis(status)

with st.expander("Simulation settings", expanded=True):
    s1, s2, s3, s4 = st.columns(4)
    horizon = s1.slider("Horizon (days)", 14, 90, 30, 7)
    crit_days = s2.number_input("Reactive: days until a Critical component fails", 1, 60, 3)
    high_days = s3.number_input("Reactive: days until a High component fails", 1, 90, 10)
    mult = s4.number_input("Reactive: unplanned repair labour multiplier", 1.0, 4.0, 1.5, 0.1)
    st.caption("Reactive-scenario failure timings are user-set illustrative assumptions, not model outputs.")

with st.spinner("Scheduling and simulating every scenario..."):
    res = availability.run_scenarios(scored, horizon=horizon, today=today,
                                     fail_days={"Critical": int(crit_days), "High": int(high_days)},
                                     repair_multiplier=float(mult))

cur = res["current"]
plan = cur["plan"]
awaiting = wos[wos["status"].isin(["Open", "Awaiting Parts", "Scheduled"])]["tail_no"].nunique()
planned = plan.dropna(subset=["start"])
planned_days = int(sum((pd.Timestamp(e) - pd.Timestamp(s)).days for s, e in zip(planned["start"], planned["end"])))
finish = planned["end"].max() if not planned.empty else today
after = cur["daily"]
after_pct = float(after[after["date"] >= finish]["availability_pct"].iloc[0]) if (after["date"] >= finish).any() else None

ui.kpi_row([
    ui.kpi("Availability now", f"{k['availability_pct']}%", f"{k['serviceable']}/{k['total']} serviceable"),
    ui.kpi("Aircraft awaiting maintenance", awaiting, "with open, waiting or scheduled work", "warn"),
    ui.kpi("Planned downtime", f"{planned_days} d", "maintenance-window days in current plan", "serious"),
    ui.kpi("After current plan completes", f"{after_pct:.1f}%" if after_pct is not None else "beyond horizon",
           f"from {finish}" if after_pct is not None else f"last job ends {finish}", "good"),
])
st.write("")

# --------------------------------------------------------------------------- comparison
st.subheader("Scenario comparison")
default = ["current", "proactive", "reactive", "expedited"]
chosen = st.multiselect("Scenarios to plot (max 4)", list(availability.SCENARIOS), default=default, max_selections=4,
                        format_func=lambda x: availability.SCENARIOS[x])
fig = go.Figure()
for i, key in enumerate(chosen):
    d = res[key]["daily"]
    fig.add_scatter(x=d["date"], y=d["availability_pct"], mode="lines", name=availability.SCENARIOS[key],
                    line=dict(width=2, color=ui.SERIES[i], shape="hv"),
                    hovertemplate="%{x|%d %b}: %{y:.1f}%<extra>" + key + "</extra>")
fig.update_layout(title=f"Daily fleet availability over {horizon} days", height=360, yaxis_title="% aircraft available",
                  yaxis_range=[max(0, min(res[c]["daily"]["availability_pct"].min() for c in chosen) - 10) if chosen else 0, 101],
                  hovermode="x unified")
st.plotly_chart(fig, config={"displayModeBar": False})

table = pd.DataFrame([{"Scenario": v["name"], **v["metrics"]} for v in res.values()])
st.dataframe(table, hide_index=True, column_config={
    "mean_availability_pct": st.column_config.NumberColumn("Mean availability %", format="%.1f"),
    "min_availability_pct": st.column_config.NumberColumn("Lowest day %", format="%.1f"),
    "end_availability_pct": st.column_config.NumberColumn("End of horizon %", format="%.1f"),
    "aircraft_days_lost": "Aircraft-days unavailable"})

# automatic, data-driven reading of the comparison
m = {kk: v["metrics"]["aircraft_days_lost"] for kk, v in res.items()}
gain_parts = m["current"] - m["expedited"]
gain_team = m["current"] - m["extra_team"]
gain_prio = m["fifo"] - m["current"]
lines = [f"- Expediting missing parts changes unavailable aircraft-days by **{-gain_parts:+d}**; "
         f"an extra team by **{-gain_team:+d}**; risk-based prioritisation vs FIFO by **{-gain_prio:+d}**."]
if gain_parts > gain_team:
    lines.append("- In this data the **binding constraint is spare-parts availability**, not team capacity.")
elif gain_team > gain_parts:
    lines.append("- In this data the **binding constraint is team capacity**, not spare-parts availability.")
else:
    lines.append("- Neither parts nor team capacity changes the outcome much at the current workload.")
diff = m["reactive"] - m["proactive"]
lines.append(f"- Acting on predictions now (proactive) vs waiting for the assumed failures (reactive): "
             f"**{diff:+d}** aircraft-days in favour of {'proactive' if diff > 0 else 'reactive' if diff < 0 else 'neither'} "
             "under the assumptions above. Change the assumptions to test sensitivity.")
st.markdown("\n".join(lines))

with st.expander("Assumptions for each scenario"):
    for key, v in res.items():
        st.markdown(f"**{v['name']}** - {v['assumptions']}")
    st.markdown(
        "**Common rules:** aircraft are unavailable during each job's maintenance window; grounding jobs keep "
        "the aircraft unavailable from today (or the failure day) until the job ends; teams work their stated "
        "hours per day and are never double-booked; jobs only start once parts are in hand. Day 0 includes "
        "downtime that starts today, so it can differ slightly from 'Availability now'.")

# --------------------------------------------------------------------------- timeline
st.subheader("Downtime timeline by aircraft")
pick = st.selectbox("Scenario", list(availability.SCENARIOS), format_func=lambda x: availability.SCENARIOS[x],
                    key="tl_scn")
p = res[pick]["plan"].copy()
if p.empty:
    ui.empty_state("No maintenance in this scenario.")
else:
    end_h = pd.Timestamp(today) + pd.Timedelta(days=horizon)
    rows = []
    for r in p.to_dict("records"):
        if r["start"] is not None and not pd.isna(r["start"]):
            rows.append({"Aircraft": r["tail_no"], "from": pd.Timestamp(r["start"]), "to": pd.Timestamp(r["end"]),
                         "Type": "Maintenance window", "job": f"{r['wo_id']} {r['component_id']} ({r['severity']})"})
        if r["grounding"]:
            g_from = pd.Timestamp(r["down_from"]) if r.get("down_from") is not None and not pd.isna(r["down_from"]) \
                else pd.Timestamp(today)
            g_to = pd.Timestamp(r["start"]) if r["start"] is not None and not pd.isna(r["start"]) else end_h
            if g_to > g_from:
                rows.append({"Aircraft": r["tail_no"], "from": g_from, "to": min(g_to, end_h),
                             "Type": "Grounded, waiting", "job": f"{r['wo_id']} - {r['reason']}"})
    tl = pd.DataFrame(rows)
    if tl.empty:
        ui.empty_state("No downtime inside the horizon.")
    else:
        tl = tl[tl["from"] < end_h]
        tl["to"] = tl["to"].clip(upper=end_h)
        fig = px.timeline(tl, x_start="from", x_end="to", y="Aircraft", color="Type", hover_data={"job": True},
                          color_discrete_map={"Maintenance window": ui.STATUS_COLORS["Under Maintenance"],
                                              "Grounded, waiting": ui.STATUS_COLORS["Unavailable"]})
        fig.update_traces(marker_line_color=ui.BG, marker_line_width=2)
        fig.add_vline(x=pd.Timestamp(today), line=dict(color=ui.ACCENT, width=1, dash="dot"))
        fig.update_layout(height=max(260, 28 * tl["Aircraft"].nunique() + 80), yaxis_title=None, xaxis_title=None,
                          xaxis_range=[pd.Timestamp(today) - pd.Timedelta(days=1), end_h],
                          title=f"{availability.SCENARIOS[pick]}")
        fig.update_yaxes(categoryorder="category descending")
        st.plotly_chart(fig, config={"displayModeBar": False})
        st.caption("Amber = aircraft in a maintenance window; red = grounded while waiting for parts, a team or "
                   "(reactive scenario) after an assumed failure.")
