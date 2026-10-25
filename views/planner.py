"""Module E - Smart Maintenance Planner."""
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from aeroready import config, db, inventory, planner, risk, ui, workorders

engine = ui.get_engine()
ui.page_header("Maintenance Planner",
               "Prioritised work-order queue, parts checks, team assignment and a transparent scheduling "
               "heuristic that never double-books a team.")
ui.show_flash()

today = date.today()
scored = ui.scored_components()
risk_now = dict(zip(scored["component_id"], scored["risk_score"]))
teams = db.table("teams")
team_names = dict(zip(teams["team_id"], teams["name"]))
wos = db.query_df("SELECT w.*, c.component_type FROM work_orders w JOIN components c USING (component_id)")
parts = db.table("spare_parts").set_index("part_number")
active = wos[wos["status"].isin(config.ACTIVE_WO_STATUSES)].copy()

ui.kpi_row([
    ui.kpi("Active work orders", len(active)),
    ui.kpi("Open (unscheduled)", int((active["status"] == "Open").sum()), "parts reserved", "accent"),
    ui.kpi("Awaiting parts", int((active["status"] == "Awaiting Parts").sum()), "blocked by stock", "serious"),
    ui.kpi("Scheduled", int((active["status"] == "Scheduled").sum()), "team & dates set", "accent"),
    ui.kpi("In progress", int((active["status"] == "In Progress").sum()), "aircraft under maintenance", "warn"),
])
st.write("")

tab_queue, tab_plan, tab_manage, tab_new = st.tabs(
    ["Priority queue", "Suggested schedule", "Manage work order", "New work order"])

# --------------------------------------------------------------------------- queue
with tab_queue:
    if active.empty:
        ui.empty_state("No active work orders. Create one from Predictive Maintenance or the Digital Twin.")
    else:
        active["current_risk"] = active["component_id"].map(risk_now)
        active["due_by"] = [workorders.due_date(s, c) for s, c in zip(active["severity"], active["created_at"])]
        active["overdue"] = (active["due_by"] < today) & (active["status"] != "In Progress")
        free = (parts["qty_on_hand"] - parts["qty_reserved"])
        active["parts"] = [
            "n/a" if not p or q == 0 else ("Reserved" if r else f"Short - {max(int(free.get(p, 0)), 0)} free")
            for p, q, r in zip(active["part_number"], active["part_qty"], active["parts_reserved"])]
        active["team"] = active["team_id"].map(team_names)
        active["_g"] = -active["grounding"]
        active["_s"] = -active["severity"].map(config.SEVERITY_RANK)
        active["_r"] = -active["current_risk"].fillna(0)
        q = active.sort_values(["_g", "_s", "_r", "created_at"]).reset_index(drop=True)
        q.insert(0, "priority", range(1, len(q) + 1))
        s1, s2 = st.columns(2)
        f_status = s1.multiselect("Status", config.ACTIVE_WO_STATUSES)
        f_sev = s2.multiselect("Severity", config.RISK_LEVELS)
        if f_status:
            q = q[q["status"].isin(f_status)]
        if f_sev:
            q = q[q["severity"].isin(f_sev)]
        st.dataframe(
            q[["priority", "wo_id", "tail_no", "component_id", "severity", "current_risk", "action", "grounding",
               "part_number", "parts", "status", "team", "scheduled_start", "scheduled_end", "due_by", "overdue",
               "recommended_action"]],
            hide_index=True, height=420,
            column_config={
                "priority": st.column_config.NumberColumn("#", width="small"), "wo_id": "WO",
                "tail_no": "Aircraft", "component_id": "Component", "severity": "Severity",
                "current_risk": st.column_config.ProgressColumn("Risk now", min_value=0, max_value=100, format="%.0f"),
                "action": "Action", "grounding": st.column_config.CheckboxColumn("Grounds a/c"),
                "part_number": "Part", "parts": "Parts check", "status": "Status", "team": "Team",
                "scheduled_start": "Start", "scheduled_end": "End (excl.)", "due_by": "Due by",
                "overdue": st.column_config.CheckboxColumn("Overdue"), "recommended_action": "Recommendation",
            })
        st.caption("Priority order: aircraft-grounding first, then severity, then current risk score, then age. "
                   "Due dates: Critical 1 day, High 3, Medium 14, Low 30 days from creation.")
        short = q[q["status"] == "Awaiting Parts"]
        for r in short.itertuples():
            eta, why = inventory.parts_ready_date(r.part_number, int(r.part_qty), today)
            st.warning(f"**{r.wo_id}** ({r.severity}, {r.tail_no}) cannot be scheduled: required spare "
                       f"**{r.part_number}** is unavailable - {why}. Earliest start {eta}.")

# --------------------------------------------------------------------------- schedule
with tab_plan:
    st.markdown(
        "The heuristic orders jobs by priority, waits for parts (purchase-order date, or lead time if nothing is on "
        "order), and gives each job to the qualified team that can **finish it earliest** without overlapping that "
        "team's other bookings. It is a greedy rule of thumb - it does **not** guarantee a globally optimal schedule.")
    replan = st.checkbox("Also re-plan work orders that are already Scheduled", value=False)
    if st.button("Generate suggested schedule", type="primary"):
        jobs = planner.load_jobs(today, keep_scheduled=not replan, risk_lookup=risk_now)
        st.session_state["plan"] = planner.build_schedule(jobs, teams, today)
        st.session_state["plan_replan"] = replan
    plan = st.session_state.get("plan")
    if plan is None:
        ui.empty_state("Generate a schedule to preview team assignments and dates before applying them.")
    elif plan.empty:
        ui.empty_state("Nothing to schedule - there are no active work orders.")
    else:
        g = plan.dropna(subset=["start"]).copy()
        g["Team"] = g["team_id"].map(team_names)
        g["start_dt"] = pd.to_datetime(g["start"])
        g["end_dt"] = pd.to_datetime(g["end"])
        g["label"] = g["wo_id"] + " " + g["tail_no"]
        fig = px.timeline(g, x_start="start_dt", x_end="end_dt", y="Team", color="severity", text="label",
                          color_discrete_map={k: ui.STATUS_COLORS[k] for k in config.RISK_LEVELS},
                          category_orders={"severity": config.RISK_LEVELS[::-1]},
                          hover_data={"wo_id": True, "component_id": True, "status": True, "reason": True,
                                      "start_dt": False, "end_dt": False, "label": False})
        fig.update_traces(marker_line_color=ui.BG, marker_line_width=2, textposition="inside",
                          insidetextanchor="start", textfont_size=10)
        fig.add_vline(x=pd.Timestamp(today), line=dict(color=ui.ACCENT, width=1, dash="dot"))
        fig.update_layout(title="Suggested team schedule (bars = maintenance windows)", height=340,
                          yaxis_title=None, xaxis_title=None, legend_title_text="Severity")
        st.plotly_chart(fig, config={"displayModeBar": False})
        show = plan.copy()
        show["team"] = show["team_id"].map(team_names)
        st.dataframe(show[["queue_rank", "wo_id", "tail_no", "component_id", "severity", "status", "team",
                           "release", "start", "end", "reason"]], hide_index=True,
                     column_config={"queue_rank": "Queue #", "release": "Parts ready", "start": "Start",
                                    "end": "End (excl.)", "reason": "Why this date"})
        overlap = planner.has_team_overlap(plan)
        st.caption("Team overlap check: " + ("**FAILED**" if overlap else "passed - no team is double-booked."))
        unq = plan[plan["team_id"].isna()]
        if not unq.empty:
            st.error(f"No qualified team for: {', '.join(unq['wo_id'])}.")
        waiting = plan[plan["status"] == "Awaiting Parts"]
        if not waiting.empty:
            st.info(f"{len(waiting)} Awaiting Parts job(s) get tentative dates only; they become Scheduled once "
                    "parts are allocated and the schedule is regenerated.")
        if st.button("Apply suggested schedule", disabled=overlap):
            try:
                n = planner.apply_schedule(plan)
            except workorders.WorkOrderError as exc:
                st.error(str(exc))
            else:
                st.session_state.pop("plan", None)
                ui.flash("success", f"Schedule applied to {n} work order(s).")
                st.rerun()

# --------------------------------------------------------------------------- manage
with tab_manage:
    manageable = wos[wos["status"].isin(config.ACTIVE_WO_STATUSES)].sort_values("created_at", ascending=False)
    if manageable.empty:
        ui.empty_state("No active work orders to manage.")
    else:
        labels = {r.wo_id: f"{r.wo_id} · {r.status} · {r.severity} · {r.title} on {r.tail_no}"
                  for r in manageable.itertuples()}
        ids = list(labels)
        pre = st.session_state.get("sel_wo")
        wo_id = st.selectbox("Work order", ids, index=ids.index(pre) if pre in ids else 0, format_func=labels.get)
        st.session_state["sel_wo"] = wo_id
        w = manageable.set_index("wo_id").loc[wo_id]
        c1, c2 = st.columns([3, 2], gap="medium")
        with c1:
            st.markdown(
                f"{ui.badge(w['status'])} {ui.badge(w['severity'])} "
                + ("<span class='ar-tag'>grounds aircraft</span>" if w["grounding"] else ""),
                unsafe_allow_html=True)
            st.markdown(
                f"**{w['title']}** - {w['component_id']} on **{w['tail_no']}**  \n"
                f"Source: {w['source']} · created {w['created_at']} · est. {w['est_hours']:.0f} labour h  \n"
                f"Risk at creation: {w['risk_score'] if pd.notna(w['risk_score']) else 'n/a'} · "
                f"risk now: {risk_now.get(w['component_id'], float('nan')):.0f}  \n"
                f"Team: {team_names.get(w['team_id'], 'unassigned')} · window: {w['scheduled_start'] or '-'} to "
                f"{w['scheduled_end'] or '-'}")
            if w["recommended_action"]:
                st.caption(f"Recommendation: {w['recommended_action']}")
            if w["notes"]:
                st.caption(f"Notes: {w['notes']}")
        with c2:
            if w["part_number"]:
                av = workorders.part_availability(w["part_number"], int(w["part_qty"]))
                st.markdown(f"**Spare part {w['part_number']}** × {w['part_qty']}  \n{av['description']}")
                if w["parts_reserved"]:
                    st.success("Parts reserved for this work order.")
                else:
                    eta, why = inventory.parts_ready_date(w["part_number"], int(w["part_qty"]), today)
                    st.warning(f"Not reserved - {av['free']} free in stock. {why.capitalize()}; earliest {eta}.")
                    if st.button("Go to Spare-Parts Intelligence"):
                        st.switch_page("views/inventory.py")
            else:
                st.info("No spare part required.")

        st.divider()
        status = w["status"]
        try:
            if status in ("Open", "Scheduled"):
                st.markdown("**Assign team and date**")
                ctype = w["component_type"]
                qualified = [t for t in teams.to_dict("records") if workorders.team_can_do(t, ctype, w["action"])]
                a1, a2, a3 = st.columns([2, 1, 1], vertical_alignment="bottom")
                team_id = a1.selectbox("Team", [t["team_id"] for t in qualified], format_func=team_names.get,
                                       key=f"as_team_{wo_id}")
                start = a2.date_input("Start", value=today, min_value=today, key=f"as_start_{wo_id}")
                if a3.button("Assign & schedule", type="primary", width="stretch"):
                    workorders.assign_and_schedule(wo_id, team_id, start)
                    ui.flash("success", f"{wo_id} scheduled with {team_names[team_id]} from {start}.")
                    st.rerun()
            x1, x2, x3, x4 = st.columns(4)
            if status in ("Open", "Scheduled") and x1.button("Start work", width="stretch",
                                                             disabled=not w["team_id"],
                                                             help="Requires an assigned team and reserved parts."):
                workorders.start_work(wo_id)
                ui.flash("success", f"{wo_id} started - {w['tail_no']} is now under maintenance.")
                st.rerun()
            if status == "Scheduled" and x2.button("Unschedule", width="stretch"):
                workorders.unschedule(wo_id)
                ui.flash("info", f"{wo_id} returned to the open queue.")
                st.rerun()
            if status == "In Progress":
                note = st.text_input("Completion notes", key=f"cn_{wo_id}", max_chars=300)
                if x1.button("Complete work order", type="primary", width="stretch"):
                    workorders.complete_work_order(wo_id, note)
                    ui.flash("success", f"{wo_id} completed: parts consumed, component condition restored, "
                                        "maintenance record written.")
                    st.rerun()
            if status in ("Open", "Awaiting Parts", "Scheduled"):
                with x4.popover("Cancel work order", width="stretch"):
                    reason = st.text_input("Reason", key=f"cr_{wo_id}")
                    if st.button("Confirm cancel", key=f"cc_{wo_id}", disabled=not reason.strip()):
                        workorders.cancel_work_order(wo_id, reason)
                        ui.flash("info", f"{wo_id} cancelled; any reserved parts released.")
                        st.rerun()
        except workorders.WorkOrderError as exc:
            st.error(str(exc))

# --------------------------------------------------------------------------- new
with tab_new:
    st.caption("Raise a work order manually (e.g. from a pilot report or inspection finding).")
    n1, n2 = st.columns(2)
    tail = n1.selectbox("Aircraft", sorted(scored["tail_no"].unique()), key="new_tail")
    cands = scored[scored["tail_no"] == tail]
    labels = {r["component_id"]: f"{ui.component_label(r)} · risk {r['risk_score']:.0f}" for r in cands.to_dict("records")}
    cid = n2.selectbox("Component", list(labels), format_func=labels.get, key="new_comp")
    comp = cands.set_index("component_id").loc[cid].to_dict() | {"component_id": cid}
    feats = risk.to_features(pd.DataFrame([comp])).iloc[0].to_dict()
    ui.work_order_panel(comp, risk.assess(engine, feats), key=f"new_{cid}", source="Manual")
