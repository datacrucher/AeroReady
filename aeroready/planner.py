"""Smart maintenance planner: a transparent greedy scheduling heuristic.

Heuristic (deliberately simple and explainable - NOT a globally optimal scheduler):
1. Order jobs by priority: aircraft-grounding first, then severity, then risk score,
   then age of the work order. (A "FIFO" ordering is also available for comparison.)
2. A job cannot start before its *release date* - today if parts are reserved, else
   the expected delivery of the part (or today + lead time if nothing is on order).
3. For each job, among the teams qualified for the component's system and action,
   pick the one that can finish it earliest; a team is never booked on overlapping
   days. Work already In Progress (and, optionally, already Scheduled) is fixed.
4. Duration = ceil(estimated labour hours / team hours per day) whole days.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date, timedelta

import pandas as pd

from . import config, db
from .inventory import parts_ready_date
from .workorders import WorkOrderError, team_can_do


@dataclass
class Job:
    wo_id: str
    tail_no: str
    component_id: str
    component_type: str
    action: str
    severity: str
    risk_score: float
    grounding: bool
    est_hours: float
    release: date
    release_reason: str
    created_at: str
    status: str = "Open"
    fixed_team: str | None = None
    fixed_start: date | None = None
    fixed_end: date | None = None
    down_from: date | None = None  # grounded from this date until the job ends (defaults to today)


def priority_key(job: Job):
    return (-int(job.grounding), -config.SEVERITY_RANK.get(job.severity, 0), -(job.risk_score or 0),
            job.created_at)


def fifo_key(job: Job):
    return (job.created_at, job.wo_id)


def earliest_slot(intervals: list[tuple[date, date]], release: date, days: int) -> date:
    """First start >= release such that [start, start+days) overlaps no interval."""
    start = release
    for s, e in sorted(intervals):
        if start + timedelta(days=days) <= s:
            break
        if e > start:
            start = e
    return start


def build_schedule(jobs: list[Job], teams: pd.DataFrame, today: date, order: str = "priority") -> pd.DataFrame:
    """Return one row per job with the assigned team and [start, end) window."""
    teams = teams.to_dict("records")
    booked: dict[str, list[tuple[date, date]]] = {t["team_id"]: [] for t in teams}
    rows = []

    fixed = [j for j in jobs if j.fixed_team and j.fixed_start]
    free = [j for j in jobs if not (j.fixed_team and j.fixed_start)]
    for j in fixed:
        end = max(j.fixed_end or today, today + timedelta(days=1)) if j.status == "In Progress" else j.fixed_end
        booked.setdefault(j.fixed_team, []).append((j.fixed_start, end))
        rows.append(_row(j, j.fixed_team, j.fixed_start, end, "fixed (already committed)"))

    key = priority_key if order == "priority" else fifo_key
    for rank, j in enumerate(sorted(free, key=key), start=1):
        best = None
        for t in teams:
            if not team_can_do(t, j.component_type, j.action):
                continue
            days = max(1, math.ceil(j.est_hours / float(t["hours_per_day"])))
            start = earliest_slot(booked[t["team_id"]], max(j.release, today), days)
            end = start + timedelta(days=days)
            load = sum((e - s).days for s, e in booked[t["team_id"]])
            cand = (end, load, t["team_id"], start)
            if best is None or cand < best:
                best = cand
        if best is None:
            rows.append(_row(j, None, None, None, "no qualified team", rank))
            continue
        end, _, team_id, start = best
        booked[team_id].append((start, end))
        wait = (start - today).days
        why = j.release_reason if j.release > today else ("team busy" if wait > 0 else "ready now")
        rows.append(_row(j, team_id, start, end, why, rank))

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["queue_rank", "start"], na_position="first").reset_index(drop=True)
    return df


def _row(j: Job, team, start, end, reason, rank=0):
    return {
        "queue_rank": rank, "wo_id": j.wo_id, "tail_no": j.tail_no, "component_id": j.component_id,
        "component_type": j.component_type, "action": j.action, "severity": j.severity,
        "risk_score": j.risk_score, "grounding": j.grounding, "status": j.status, "team_id": team,
        "release": j.release, "start": start, "end": end, "reason": reason,
        "down_from": j.down_from,
    }


def has_team_overlap(plan: pd.DataFrame) -> bool:
    """True if any team has two overlapping windows (used by tests and as a safety check)."""
    p = plan.dropna(subset=["team_id", "start"])
    for _, g in p.groupby("team_id"):
        spans = sorted(zip(g["start"], g["end"]))
        if any(spans[i][1] > spans[i + 1][0] for i in range(len(spans) - 1)):
            return True
    return False


# --------------------------------------------------------------------------- DB integration
def load_jobs(today: date | None = None, keep_scheduled: bool = True, risk_lookup: dict | None = None) -> list[Job]:
    today = today or date.today()
    wos = db.query_df(
        "SELECT w.*, c.component_type FROM work_orders w JOIN components c USING (component_id) "
        "WHERE w.status IN ('Open','Awaiting Parts','Scheduled','In Progress')")
    jobs = []
    for r in wos.to_dict("records"):
        if r["status"] == "Awaiting Parts":
            release, reason = parts_ready_date(r["part_number"], int(r["part_qty"]), today)
        else:
            release, reason = today, "parts reserved"
        fixed = r["status"] == "In Progress" or (keep_scheduled and r["status"] == "Scheduled")
        risk = r["risk_score"]
        if risk_lookup is not None and r["component_id"] in risk_lookup:
            risk = risk_lookup[r["component_id"]]
        jobs.append(Job(
            wo_id=r["wo_id"], tail_no=r["tail_no"], component_id=r["component_id"],
            component_type=r["component_type"], action=r["action"], severity=r["severity"],
            risk_score=float(risk) if pd.notna(risk) else 0.0, grounding=bool(r["grounding"]),
            est_hours=float(r["est_hours"]), release=release, release_reason=reason,
            created_at=r["created_at"], status=r["status"],
            fixed_team=r["team_id"] if fixed and r["scheduled_start"] else None,
            fixed_start=date.fromisoformat(r["scheduled_start"]) if fixed and r["scheduled_start"] else None,
            fixed_end=date.fromisoformat(r["scheduled_end"]) if fixed and r["scheduled_end"] else None,
        ))
    return jobs


def apply_schedule(plan: pd.DataFrame) -> int:
    """Persist a plan. Parts-ready jobs become Scheduled; Awaiting Parts jobs keep their
    status but store the tentative team and dates. Returns number of work orders updated."""
    if has_team_overlap(plan):
        raise WorkOrderError("Plan contains overlapping team bookings; not applied.")
    n = 0
    with db.connect() as conn:
        for r in plan.to_dict("records"):
            if r["reason"].startswith("fixed") or r["team_id"] is None or pd.isna(r["start"]):
                continue
            status = "Scheduled" if r["status"] in ("Open", "Scheduled") else r["status"]
            conn.execute(
                "UPDATE work_orders SET team_id = ?, scheduled_start = ?, scheduled_end = ?, status = ?, "
                "updated_at = ? WHERE wo_id = ?",
                (r["team_id"], r["start"].isoformat(), r["end"].isoformat(), status, db.now_iso(), r["wo_id"]))
            n += 1
        db.log_activity(conn, "Planning", f"Suggested schedule applied to {n} work order(s)")
    return n


def with_release_today(jobs: list[Job], today: date) -> list[Job]:
    return [replace(j, release=today, release_reason="parts expedited (scenario)") for j in jobs]
