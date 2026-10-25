"""Fleet availability: current status derivation and the scenario simulator.

Aircraft status rule (used everywhere in the app):
* "Under Maintenance" - the aircraft has a work order In Progress.
* "Unavailable"       - otherwise, it has an active work order flagged as grounding
                        (e.g. a Critical finding waiting for parts or a team).
* "Serviceable"       - neither of the above.
Fleet availability % = serviceable aircraft / total aircraft.

Simulator rule: on day d an aircraft is unavailable if d falls inside the [start, end)
window of any of its scheduled jobs, or if it has a grounding job and
down_from <= d < end (an unscheduled grounding job grounds it for the whole horizon).
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import config, db
from .planner import Job, build_schedule, load_jobs
from .workorders import default_hours, default_part


# --------------------------------------------------------------------------- current status
def aircraft_status(aircraft: pd.DataFrame, work_orders: pd.DataFrame) -> pd.DataFrame:
    active = work_orders[work_orders["status"].isin(config.ACTIVE_WO_STATUSES)]
    in_prog = active[active["status"] == "In Progress"]
    grounded = active[(active["grounding"] == 1) & (active["status"] != "In Progress")]
    status, reason = [], []
    for tail in aircraft["tail_no"]:
        ip = in_prog[in_prog["tail_no"] == tail]
        gr = grounded[grounded["tail_no"] == tail]
        if len(ip):
            status.append("Under Maintenance")
            reason.append("In progress: " + ", ".join(ip["wo_id"]))
        elif len(gr):
            status.append("Unavailable")
            reason.append("Grounded pending " + ", ".join(f"{w} ({s})" for w, s in zip(gr["wo_id"], gr["status"])))
        else:
            status.append("Serviceable")
            reason.append("")
    out = aircraft.copy()
    out["status"] = status
    out["status_reason"] = reason
    open_counts = active.groupby("tail_no").size()
    out["active_work_orders"] = out["tail_no"].map(open_counts).fillna(0).astype(int)
    return out


def fleet_kpis(status_df: pd.DataFrame) -> dict:
    total = len(status_df)
    counts = status_df["status"].value_counts()
    serviceable = int(counts.get("Serviceable", 0))
    return {
        "total": total,
        "serviceable": serviceable,
        "under_maintenance": int(counts.get("Under Maintenance", 0)),
        "unavailable": int(counts.get("Unavailable", 0)),
        "availability_pct": round(100 * serviceable / total, 1) if total else 0.0,
    }


# --------------------------------------------------------------------------- simulation
def daily_availability(plan: pd.DataFrame, tails: list[str], today: date, horizon: int) -> pd.DataFrame:
    days = [today + timedelta(days=i) for i in range(horizon)]
    down = {t: np.zeros(horizon, dtype=bool) for t in tails}
    for r in plan.to_dict("records"):
        if r["tail_no"] not in down:
            continue
        mask = down[r["tail_no"]]
        start, end = r["start"], r["end"]
        scheduled = start is not None and not pd.isna(start)
        if scheduled:
            for i, d in enumerate(days):
                if start <= d < end:
                    mask[i] = True
        if r["grounding"]:
            from_day = r.get("down_from") if r.get("down_from") is not None and not pd.isna(r.get("down_from")) else today
            until = end if scheduled else days[-1] + timedelta(days=1)
            for i, d in enumerate(days):
                if from_day <= d < until:
                    mask[i] = True
    available = [sum(not down[t][i] for t in tails) for i in range(horizon)]
    return pd.DataFrame({
        "date": days,
        "available": available,
        "availability_pct": [round(100 * a / len(tails), 1) if tails else 0 for a in available],
    })


def summarise(daily: pd.DataFrame, n_aircraft: int) -> dict:
    return {
        "mean_availability_pct": round(float(daily["availability_pct"].mean()), 1),
        "min_availability_pct": float(daily["availability_pct"].min()),
        "end_availability_pct": float(daily["availability_pct"].iloc[-1]),
        "aircraft_days_lost": int(n_aircraft * len(daily) - daily["available"].sum()),
    }


def _unaddressed_jobs(scored: pd.DataFrame, today: date, mode: str, fail_days: dict,
                      repair_multiplier: float, stock: dict) -> list[Job]:
    """Synthetic jobs for High/Critical components that have no active work order.

    mode="proactive": planned now using the recommended action (Critical grounds now).
    mode="reactive":  no action until an assumed failure ``fail_days[level]`` days out,
                      then an unplanned replacement grounds the aircraft until repaired.
    ``stock`` (part -> free qty) is consumed as parts are claimed.
    """
    active = db.query_df("SELECT component_id FROM work_orders WHERE status IN "
                         "('Open','Awaiting Parts','Scheduled','In Progress')")
    risky = scored[scored["risk_level"].isin(["High", "Critical"])
                   & ~scored["component_id"].isin(active["component_id"])]
    parts = db.table("spare_parts").set_index("part_number")
    jobs = []
    for r in risky.sort_values("risk_score", ascending=False).itertuples():
        if mode == "proactive":
            action = r.recommended_action or "Inspect & Service"
            event_day = today
            hours = default_hours(r.component_type, action)
            grounding = r.risk_level == "Critical"
        else:
            action = "Replace"
            event_day = today + timedelta(days=int(fail_days[r.risk_level]))
            hours = default_hours(r.component_type, action) * repair_multiplier
            grounding = True
        part = default_part(r.component_type, action)
        if stock.get(part, 0) >= 1:
            stock[part] -= 1
            release, why = event_day, "parts in stock"
        else:
            release = event_day + timedelta(days=int(parts.loc[part, "lead_time_days"]))
            why = f"part {part} not in stock - {int(parts.loc[part, 'lead_time_days'])}-day lead time"
        jobs.append(Job(
            wo_id=f"SIM-{r.component_id}", tail_no=r.tail_no, component_id=r.component_id,
            component_type=r.component_type, action=action, severity=r.risk_level,
            risk_score=float(r.risk_score), grounding=grounding, est_hours=hours, release=release,
            release_reason=why, created_at="9999", status="Proposed",
            down_from=event_day if grounding else None))
    return jobs


SCENARIOS = {
    "current": "Current plan",
    "proactive": "Proactive: act on all High/Critical predictions now",
    "reactive": "Reactive: wait for failures of unaddressed High/Critical components",
    "expedited": "Current plan + all missing parts expedited",
    "extra_team": "Current plan + one extra general-purpose team",
    "fifo": "Current work orders, first-in-first-out (no risk prioritisation)",
}

ASSUMPTIONS = {
    "current": "Existing active work orders only, scheduled with the priority heuristic. Parts arrive on "
               "the purchase-order date, or after the lead time if nothing is on order. Unaddressed "
               "predictions cause no downtime inside the horizon (optimistic).",
    "proactive": "Current plan, plus a planned work order now for every High/Critical component without one, "
                 "using its recommended action. Critical findings ground the aircraft until the work is done.",
    "reactive": "Current plan, but unaddressed High/Critical components get no preventive action. They are "
                "assumed to fail after the configured number of days and need an unplanned replacement whose "
                "labour is multiplied by the configured factor. The aircraft is grounded from failure to repair. "
                "The failure timings are illustrative assumptions, not predictions.",
    "expedited": "Same as current plan, but every Awaiting Parts work order is assumed to have its parts today.",
    "extra_team": "Same as current plan with one additional fictional team qualified for all systems and actions.",
    "fifo": "Same work orders as the current plan, ordered by creation time instead of risk/severity.",
}


def run_scenarios(scored: pd.DataFrame, horizon: int = 30, today: date | None = None,
                  fail_days: dict | None = None, repair_multiplier: float = 1.5,
                  keys: list[str] | None = None) -> dict:
    today = today or date.today()
    fail_days = fail_days or {"Critical": 3, "High": 10}
    teams = db.table("teams")
    tails = db.table("aircraft")["tail_no"].tolist()
    risk_lookup = dict(zip(scored["component_id"], scored["risk_score"])) if not scored.empty else {}
    base_jobs = load_jobs(today, keep_scheduled=True, risk_lookup=risk_lookup)
    parts = db.table("spare_parts")

    def free_stock():
        return dict(zip(parts["part_number"], (parts["qty_on_hand"] - parts["qty_reserved"]).clip(lower=0)))

    results = {}
    for key in keys or list(SCENARIOS):
        jobs, t, order = list(base_jobs), teams, "priority"
        if key == "proactive":
            jobs += _unaddressed_jobs(scored, today, "proactive", fail_days, repair_multiplier, free_stock())
        elif key == "reactive":
            jobs += _unaddressed_jobs(scored, today, "reactive", fail_days, repair_multiplier, free_stock())
        elif key == "expedited":
            jobs = [j if j.status != "Awaiting Parts" else replace(j, release=today,
                                                                   release_reason="parts expedited (scenario)")
                    for j in jobs]
        elif key == "extra_team":
            t = pd.concat([teams, pd.DataFrame([{
                "team_id": "T-SIM", "name": "Additional team (scenario)",
                "systems": "Propulsion,Hydraulics,Electrical", "hours_per_day": 8, "service_only": 0}])],
                ignore_index=True)
        elif key == "fifo":
            order = "fifo"
        plan = build_schedule(jobs, t, today, order=order)
        daily = daily_availability(plan, tails, today, horizon)
        results[key] = {
            "name": SCENARIOS[key], "assumptions": ASSUMPTIONS[key], "plan": plan, "daily": daily,
            "metrics": summarise(daily, len(tails)),
        }
    return results
