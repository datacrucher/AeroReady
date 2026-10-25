"""Work-order lifecycle with consistent spare-parts reservation.

Invariants maintained here:
* ``spare_parts.qty_reserved`` == sum of ``part_qty`` over active work orders with
  ``parts_reserved = 1``.
* A work order whose parts cannot be reserved is in status "Awaiting Parts" and
  cannot be scheduled or started until stock arrives (see :func:`allocate_parts`).
* Completing a work order consumes the reserved stock, writes a maintenance record,
  closes the component's open faults and restores its simulated condition.
* A team cannot be assigned two work orders whose dates overlap.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta

from . import config, db

SEVERITY_DUE_DAYS = {"Critical": 1, "High": 3, "Medium": 14, "Low": 30}

TRANSITIONS = {
    "Open": {"Scheduled", "In Progress", "Cancelled"},
    "Awaiting Parts": {"Cancelled"},
    "Scheduled": {"Open", "In Progress", "Cancelled"},
    "In Progress": {"Completed"},
    "Completed": set(),
    "Cancelled": set(),
}


class WorkOrderError(ValueError):
    """Raised for invalid work-order operations; message is user-facing."""


# --------------------------------------------------------------------------- helpers
def default_part(component_type: str, action: str) -> str:
    spec = config.COMPONENT_TYPES[component_type]
    return spec["replace_part"] if action == "Replace" else spec["service_part"]


def default_hours(component_type: str, action: str) -> float:
    spec = config.COMPONENT_TYPES[component_type]
    return float(spec["replace_hours"] if action == "Replace" else spec["service_hours"])


def part_availability(part_number: str, qty: int = 1) -> dict:
    part = db.query_one("SELECT * FROM spare_parts WHERE part_number = ?", (part_number,))
    if part is None:
        raise WorkOrderError(f"Unknown part number '{part_number}'.")
    po = db.query_one(
        "SELECT po_id, qty, expected_on FROM purchase_orders WHERE part_number = ? AND status = 'Ordered' "
        "ORDER BY expected_on LIMIT 1", (part_number,))
    free = part["qty_on_hand"] - part["qty_reserved"]
    return {
        **part,
        "free": free,
        "requested": qty,
        "sufficient": free >= qty,
        "open_po": po,
    }


def _get(conn, wo_id: str) -> dict:
    row = conn.execute("SELECT * FROM work_orders WHERE wo_id = ?", (wo_id,)).fetchone()
    if row is None:
        raise WorkOrderError(f"Work order {wo_id} not found.")
    return dict(row)


def _reserve(conn, part_number: str, qty: int) -> bool:
    cur = conn.execute(
        "UPDATE spare_parts SET qty_reserved = qty_reserved + ? "
        "WHERE part_number = ? AND qty_on_hand - qty_reserved >= ?", (qty, part_number, qty))
    return cur.rowcount == 1


def _release(conn, wo: dict) -> None:
    if wo["parts_reserved"] and wo["part_number"]:
        conn.execute("UPDATE spare_parts SET qty_reserved = MAX(qty_reserved - ?, 0) WHERE part_number = ?",
                     (wo["part_qty"], wo["part_number"]))


def _touch(conn, wo_id: str, **fields) -> None:
    fields["updated_at"] = db.now_iso()
    sets = ", ".join(f"{k} = ?" for k in fields)
    conn.execute(f"UPDATE work_orders SET {sets} WHERE wo_id = ?", (*fields.values(), wo_id))


# --------------------------------------------------------------------------- create
def create_work_order(
    tail_no: str,
    component_id: str,
    action: str,
    severity: str,
    *,
    part_number: str | None = None,
    part_qty: int = 1,
    est_hours: float | None = None,
    grounding: bool = False,
    risk_score: float | None = None,
    recommended_action: str | None = None,
    source: str = "Manual",
    notes: str = "",
) -> dict:
    """Create a work order, reserving parts if available. Returns a summary dict."""
    if action not in config.WO_ACTIONS:
        raise WorkOrderError(f"Action must be one of {config.WO_ACTIONS}.")
    if severity not in config.SEVERITY_RANK:
        raise WorkOrderError(f"Severity must be one of {list(config.SEVERITY_RANK)}.")
    if part_qty is None or int(part_qty) < 0:
        raise WorkOrderError("Part quantity must be zero or a positive integer.")
    part_qty = int(part_qty)

    with db.connect() as conn:
        comp = conn.execute("SELECT * FROM components WHERE component_id = ?", (component_id,)).fetchone()
        if comp is None:
            raise WorkOrderError(f"Component {component_id} not found.")
        if comp["tail_no"] != tail_no:
            raise WorkOrderError(f"Component {component_id} is not fitted to aircraft {tail_no}.")
        dup = conn.execute(
            f"SELECT wo_id FROM work_orders WHERE component_id = ? AND status IN "
            f"({','.join('?' * len(config.ACTIVE_WO_STATUSES))})",
            (component_id, *config.ACTIVE_WO_STATUSES)).fetchone()
        if dup:
            raise WorkOrderError(f"Component {component_id} already has an active work order ({dup[0]}).")

        part_number = part_number or default_part(comp["component_type"], action)
        if part_qty > 0 and conn.execute("SELECT 1 FROM spare_parts WHERE part_number = ?",
                                         (part_number,)).fetchone() is None:
            raise WorkOrderError(f"Unknown part number '{part_number}'.")
        hours = float(est_hours) if est_hours else default_hours(comp["component_type"], action)
        if hours <= 0:
            raise WorkOrderError("Estimated labour hours must be positive.")

        reserved = part_qty == 0 or _reserve(conn, part_number, part_qty)
        status = "Open" if reserved else "Awaiting Parts"
        wo_id = db.next_id(conn, "work_orders", "wo_id", "WO-")
        now = db.now_iso()
        conn.execute(
            "INSERT INTO work_orders (wo_id, tail_no, component_id, title, action, severity, risk_score, "
            "recommended_action, part_number, part_qty, parts_reserved, status, team_id, scheduled_start, "
            "scheduled_end, est_hours, grounding, source, created_at, updated_at, completed_at, notes) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,NULL,NULL,NULL,?,?,?,?,?,NULL,?)",
            (wo_id, tail_no, component_id, f"{action}: {comp['component_type']} {comp['position']}", action,
             severity, risk_score, recommended_action, part_number if part_qty else None, part_qty,
             int(reserved and part_qty > 0), status, hours, int(grounding), source, now, now, notes))
        msg = (f"{wo_id} created for {component_id} ({severity}); "
               + ("parts reserved." if reserved and part_qty else
                  "no parts required." if not part_qty else f"{part_number} not in stock - awaiting parts."))
        db.log_activity(conn, "Work order", msg, tail_no)
    return {"wo_id": wo_id, "status": status, "parts_reserved": reserved, "part_number": part_number,
            "message": msg}


# --------------------------------------------------------------------------- assignment
def team_conflicts(conn, team_id: str, start: date, end: date, exclude_wo: str | None = None) -> list[str]:
    """Active work orders of ``team_id`` whose [start, end) window overlaps [start, end)."""
    rows = conn.execute(
        "SELECT wo_id, scheduled_start, scheduled_end FROM work_orders WHERE team_id = ? "
        "AND status IN ('Scheduled','In Progress') AND scheduled_start IS NOT NULL AND wo_id != ?",
        (team_id, exclude_wo or "")).fetchall()
    out = []
    for r in rows:
        s, e = date.fromisoformat(r["scheduled_start"]), date.fromisoformat(r["scheduled_end"])
        if s < end and start < e:
            out.append(r["wo_id"])
    return out


def team_can_do(team: dict, component_type: str, action: str) -> bool:
    system = config.COMPONENT_TYPES[component_type]["system"]
    if system not in team["systems"].split(","):
        return False
    return not (team["service_only"] and action == "Replace")


def assign_and_schedule(wo_id: str, team_id: str, start: date, duration_days: int | None = None) -> dict:
    """Manually assign a team and dates. Rejects skill mismatches and team overlaps."""
    with db.connect() as conn:
        wo = _get(conn, wo_id)
        if wo["status"] not in ("Open", "Scheduled"):
            raise WorkOrderError(
                f"{wo_id} is '{wo['status']}'; only Open or Scheduled work orders can be scheduled"
                + (" (parts must be available first)." if wo["status"] == "Awaiting Parts" else "."))
        team = conn.execute("SELECT * FROM teams WHERE team_id = ?", (team_id,)).fetchone()
        if team is None:
            raise WorkOrderError(f"Team {team_id} not found.")
        ctype = conn.execute("SELECT component_type FROM components WHERE component_id = ?",
                             (wo["component_id"],)).fetchone()[0]
        if not team_can_do(dict(team), ctype, wo["action"]):
            raise WorkOrderError(f"{team['name']} is not qualified for '{wo['action']}' on {ctype}.")
        days = duration_days or max(1, math.ceil(wo["est_hours"] / team["hours_per_day"]))
        end = start + timedelta(days=days)
        clash = team_conflicts(conn, team_id, start, end, exclude_wo=wo_id)
        if clash:
            raise WorkOrderError(f"{team['name']} is already booked on {', '.join(clash)} in that window.")
        _touch(conn, wo_id, team_id=team_id, scheduled_start=start.isoformat(),
               scheduled_end=end.isoformat(), status="Scheduled")
        db.log_activity(conn, "Planning", f"{wo_id} scheduled with {team['name']} "
                                          f"{start.isoformat()} to {end.isoformat()}", wo["tail_no"])
    return {"wo_id": wo_id, "start": start, "end": end}


# --------------------------------------------------------------------------- transitions
def start_work(wo_id: str) -> None:
    with db.connect() as conn:
        wo = _get(conn, wo_id)
        if "In Progress" not in TRANSITIONS[wo["status"]]:
            raise WorkOrderError(f"Cannot start {wo_id} from status '{wo['status']}'.")
        if wo["part_qty"] and not wo["parts_reserved"]:
            raise WorkOrderError(f"Parts for {wo_id} are not reserved.")
        if not wo["team_id"]:
            raise WorkOrderError(f"Assign a team to {wo_id} before starting work.")
        today = date.today()
        start = date.fromisoformat(wo["scheduled_start"]) if wo["scheduled_start"] else today
        days = max(1, math.ceil(wo["est_hours"] / 8))
        if start > today or not wo["scheduled_end"]:
            start = today
        end = max(date.fromisoformat(wo["scheduled_end"]) if wo["scheduled_end"] else today,
                  start + timedelta(days=days))
        _touch(conn, wo_id, status="In Progress", scheduled_start=start.isoformat(),
               scheduled_end=end.isoformat())
        db.log_activity(conn, "Work order", f"{wo_id} started", wo["tail_no"])


def unschedule(wo_id: str) -> None:
    with db.connect() as conn:
        wo = _get(conn, wo_id)
        if wo["status"] != "Scheduled":
            raise WorkOrderError(f"{wo_id} is not scheduled.")
        _touch(conn, wo_id, status="Open", team_id=None, scheduled_start=None, scheduled_end=None)
        db.log_activity(conn, "Planning", f"{wo_id} unscheduled", wo["tail_no"])


def cancel_work_order(wo_id: str, reason: str = "") -> None:
    with db.connect() as conn:
        wo = _get(conn, wo_id)
        if "Cancelled" not in TRANSITIONS[wo["status"]]:
            raise WorkOrderError(f"Cannot cancel {wo_id} from status '{wo['status']}'.")
        _release(conn, wo)
        _touch(conn, wo_id, status="Cancelled", parts_reserved=0,
               notes=((wo["notes"] or "") + f" Cancelled: {reason}").strip())
        db.log_activity(conn, "Work order", f"{wo_id} cancelled", wo["tail_no"])
    allocate_parts()


def complete_work_order(wo_id: str, notes: str = "") -> None:
    """Consume parts, record maintenance and restore the component's simulated condition."""
    with db.connect() as conn:
        wo = _get(conn, wo_id)
        if wo["status"] != "In Progress":
            raise WorkOrderError(f"Only In Progress work orders can be completed ({wo_id} is '{wo['status']}').")
        if wo["parts_reserved"] and wo["part_number"]:
            conn.execute(
                "UPDATE spare_parts SET qty_on_hand = MAX(qty_on_hand - ?, 0), "
                "qty_reserved = MAX(qty_reserved - ?, 0) WHERE part_number = ?",
                (wo["part_qty"], wo["part_qty"], wo["part_number"]))
        comp = dict(conn.execute("SELECT * FROM components WHERE component_id = ?",
                                 (wo["component_id"],)).fetchone())
        if wo["action"] == "Replace":
            new = dict(operating_hours=0.0, age_years=0.0, prior_faults=0, temp_deviation=1.0,
                       vibration=1.6, pressure_deviation=1.0)
        else:  # servicing removes most, not all, of the deviation
            new = dict(operating_hours=comp["operating_hours"], age_years=comp["age_years"],
                       prior_faults=comp["prior_faults"],
                       temp_deviation=round(1.0 + (comp["temp_deviation"] - 1.0) * 0.25, 1),
                       vibration=round(1.6 + max(comp["vibration"] - 1.6, 0) * 0.25, 2),
                       pressure_deviation=round(1.0 + max(comp["pressure_deviation"] - 1.0, 0) * 0.25, 1))
        now = db.now_iso()
        conn.execute(
            "UPDATE components SET operating_hours = ?, age_years = ?, prior_faults = ?, temp_deviation = ?, "
            "vibration = ?, pressure_deviation = ?, hours_since_maintenance = 0, "
            "maintenance_count = maintenance_count + 1, last_reading_at = ? WHERE component_id = ?",
            (new["operating_hours"], new["age_years"], new["prior_faults"], new["temp_deviation"],
             new["vibration"], new["pressure_deviation"], now, wo["component_id"]))
        conn.execute(
            "INSERT INTO sensor_readings (component_id, recorded_at, operating_hours, hours_since_maintenance, "
            "temp_deviation, vibration, pressure_deviation, source) VALUES (?,?,?,?,?,?,?,?)",
            (wo["component_id"], now, new["operating_hours"], 0.0, new["temp_deviation"], new["vibration"],
             new["pressure_deviation"], "post-maintenance"))
        conn.execute(
            "INSERT INTO maintenance_records (component_id, tail_no, performed_on, record_type, description, "
            "hours_at, team_id, work_order_id) VALUES (?,?,?,?,?,?,?,?)",
            (wo["component_id"], wo["tail_no"], date.today().isoformat(),
             "Unscheduled" if wo["source"] == "Predictive" else "Scheduled",
             f"{wo_id} completed: {wo['action'].lower()}" + (f" - {notes}" if notes else ""),
             comp["operating_hours"], wo["team_id"], wo_id))
        conn.execute("UPDATE faults SET status = 'Closed' WHERE component_id = ? AND status = 'Open'",
                     (wo["component_id"],))
        _touch(conn, wo_id, status="Completed", completed_at=now, parts_reserved=0,
               scheduled_end=date.today().isoformat(),  # actual (exclusive) end of downtime
               notes=((wo["notes"] or "") + (f" {notes}" if notes else "")).strip())
        db.log_activity(conn, "Work order", f"{wo_id} completed on {wo['component_id']}", wo["tail_no"])


def allocate_parts() -> list[str]:
    """Reserve newly available stock for Awaiting Parts work orders, highest priority first."""
    allocated = []
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT * FROM work_orders WHERE status = 'Awaiting Parts' ORDER BY grounding DESC, "
            "CASE severity WHEN 'Critical' THEN 4 WHEN 'High' THEN 3 WHEN 'Medium' THEN 2 ELSE 1 END DESC, "
            "created_at").fetchall()
        for wo in rows:
            if _reserve(conn, wo["part_number"], wo["part_qty"]):
                _touch(conn, wo["wo_id"], status="Open", parts_reserved=1)
                db.log_activity(conn, "Parts", f"{wo['part_number']} allocated to {wo['wo_id']}", wo["tail_no"])
                allocated.append(wo["wo_id"])
    return allocated


def due_date(severity: str, created_at: str) -> date:
    created = datetime.fromisoformat(created_at).date()
    return created + timedelta(days=SEVERITY_DUE_DAYS.get(severity, 30))


# --------------------------------------------------------------------------- telemetry
def record_reading(component_id: str, temp_deviation: float, vibration: float, pressure_deviation: float,
                   source: str = "digital-twin") -> None:
    """Persist a sensor reading and make it the component's current condition."""
    for name, val in (("temp_deviation", temp_deviation), ("vibration", vibration),
                      ("pressure_deviation", pressure_deviation)):
        lo, hi, _ = config.FEATURE_RANGES[name]
        if val is None or not (lo <= float(val) <= hi):
            raise WorkOrderError(f"{config.FEATURE_LABELS[name]} must be between {lo} and {hi}.")
    with db.connect() as conn:
        comp = conn.execute("SELECT * FROM components WHERE component_id = ?", (component_id,)).fetchone()
        if comp is None:
            raise WorkOrderError(f"Component {component_id} not found.")
        now = db.now_iso()
        conn.execute(
            "INSERT INTO sensor_readings (component_id, recorded_at, operating_hours, hours_since_maintenance, "
            "temp_deviation, vibration, pressure_deviation, source) VALUES (?,?,?,?,?,?,?,?)",
            (component_id, now, comp["operating_hours"], comp["hours_since_maintenance"],
             float(temp_deviation), float(vibration), float(pressure_deviation), source))
        conn.execute(
            "UPDATE components SET temp_deviation = ?, vibration = ?, pressure_deviation = ?, last_reading_at = ? "
            "WHERE component_id = ?", (float(temp_deviation), float(vibration), float(pressure_deviation), now,
                                       component_id))
        db.log_activity(conn, "Telemetry", f"New {source} reading committed for {component_id}", comp["tail_no"])
