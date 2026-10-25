"""Reproducible synthetic demonstration dataset (entirely fictional).

The fleet is mostly healthy, with a handful of deliberately degraded components so
the demo has a story to tell:

* KM-303 engine replacement in progress            -> aircraft under maintenance
* FT-205 generator service in progress             -> aircraft under maintenance
* AC-402 hydraulic pump critical, part out of stock -> aircraft unavailable (grounded)
* HU-504 landing-gear actuator inspection scheduled
* Several degraded components with no work order yet (e.g. KM-306 fuel pump,
  whose replacement part is out of stock) for the predictive workflow to pick up.
"""
from __future__ import annotations

import math
from datetime import date, datetime, timedelta

import numpy as np

from . import config, db
from .model import simulate_sensors

HISTORY_DAYS = 30

# (tail_no, component_type, position) -> forced hidden degradation, plus usage overrides
FORCED = {
    ("KM-303", "Gas Turbine Engine", "#1"): dict(degr=0.85, hsm=780, faults=3),
    ("FT-205", "Generator", "#1"): dict(degr=0.55, hsm=640, faults=2),
    ("AC-402", "Hydraulic Pump", "#1"): dict(degr=0.9, hsm=900, faults=4),
    ("HU-504", "Landing Gear Actuator", "#1"): dict(degr=0.6, hsm=700, faults=2),
    ("KM-306", "Fuel Pump", "#1"): dict(degr=0.88, hsm=950, faults=4, life=104),
    ("FT-202", "Gas Turbine Engine", "#1"): dict(degr=0.7, hsm=820, faults=3),
    ("AC-405", "Auxiliary Power Unit", "#1"): dict(degr=0.65, hsm=760, faults=2),
    ("HU-501", "Generator", "#1"): dict(degr=0.5, hsm=600, faults=2),
    ("AC-404", "Gas Turbine Engine", "#2"): dict(degr=0.45, hsm=520, faults=1),
}

FAULT_CODES = {
    "vibration": ("FC-VIB-01", "Vibration exceedance recorded by health monitoring unit"),
    "temp": ("FC-TMP-02", "Over-temperature indication during ground run"),
    "pressure": ("FC-PRS-03", "Pressure fluctuation outside nominal band"),
    "bite": ("FC-BIT-04", "Built-in test intermittent fail"),
    "leak": ("FC-LKG-05", "Minor fluid seepage observed at inspection"),
}

MAINT_DESCRIPTIONS = {
    "Scheduled": "Scheduled servicing completed per fictional maintenance programme",
    "Inspection": "Condition inspection - no defect found",
    "Unscheduled": "Unscheduled rectification following fault report",
}


def _d(day: date) -> str:
    return day.isoformat()


def _dt(day: date, hour: int = 9) -> str:
    return datetime(day.year, day.month, day.day, hour).isoformat(sep=" ")


def seed(engine=None, seed_value: int = config.RANDOM_SEED, today: date | None = None) -> None:
    """Drop and recreate all tables with a fresh fictional dataset."""
    rng = np.random.default_rng(seed_value)
    today = today or date.today()
    db.drop_all()
    db.init_schema()

    aircraft_rows, comp_rows, reading_rows, record_rows, fault_rows = [], [], [], [], []
    utilisation = {}  # tail -> flying hours per day

    for type_idx, (ac_type, spec) in enumerate(config.AIRCRAFT_TYPES.items()):
        for n in range(1, 7):
            tail = f"{spec['prefix']}-{(type_idx + 2) * 100 + n}"
            year = int(rng.integers(2004, 2021))
            fh = float(round(rng.uniform(900, 380 * (2026 - year) + 1500), 0))
            aircraft_rows.append((tail, ac_type, config.BASES[(n - 1) % 3],
                                  f"No. {type_idx + 1}0{(n - 1) // 3 + 1} Demo Squadron", year, fh))
            utilisation[tail] = float(rng.uniform(1.0, 2.6))

            engines = 2 if "Albatross" in ac_type else 1
            layout = [("Gas Turbine Engine", f"#{i}") for i in range(1, engines + 1)]
            layout += [(c, "#1") for c in config.COMPONENT_TYPES if c != "Gas Turbine Engine"]
            for ctype, pos in layout:
                comp_rows.append(_component(rng, tail, year, ctype, pos, utilisation[tail], today,
                                            reading_rows, record_rows, fault_rows))

    with db.connect() as conn:
        conn.executemany("INSERT INTO aircraft VALUES (?,?,?,?,?,?)", aircraft_rows)
        conn.executemany(
            "INSERT INTO components VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", comp_rows)
        conn.executemany(
            "INSERT INTO sensor_readings (component_id, recorded_at, operating_hours, "
            "hours_since_maintenance, temp_deviation, vibration, pressure_deviation, source) "
            "VALUES (?,?,?,?,?,?,?,?)", reading_rows)
        conn.executemany(
            "INSERT INTO maintenance_records (component_id, tail_no, performed_on, record_type, "
            "description, hours_at, team_id, work_order_id) VALUES (?,?,?,?,?,?,?,?)", record_rows)
        conn.executemany("INSERT INTO faults VALUES (?,?,?,?,?,?,?,?)", fault_rows)
        conn.executemany("INSERT INTO teams VALUES (?,?,?,?,?)", config.TEAMS)
        conn.executemany("INSERT INTO spare_parts VALUES (?,?,?,?,?,?,?,?,?,?)", _parts(rng))
        conn.executemany(
            "INSERT INTO maintenance_schedule VALUES (?,?,?,?,?)",
            _schedule(rng, aircraft_rows))
        _purchase_orders(conn, today)
        _work_orders(conn, rng, today, engine)
        conn.execute("INSERT INTO meta VALUES ('seeded_at', ?)", (db.now_iso(),))
        conn.execute("INSERT INTO meta VALUES ('seed', ?)", (str(seed_value),))
        db.log_activity(conn, "System", "Demonstration dataset generated (fictional, synthetic).")


def ensure_seeded(engine=None) -> bool:
    """Seed on first launch. Returns True if a new dataset was generated."""
    db.init_schema()
    if db.is_seeded():
        return False
    seed(engine)
    return True


# --------------------------------------------------------------------------- builders
def _component(rng, tail, year, ctype, pos, util, today, readings, records, faults):
    spec = config.COMPONENT_TYPES[ctype]
    forced = FORCED.get((tail, ctype, pos), {})
    comp_id = f"{tail}-{''.join(w[0] for w in ctype.split())}{pos[1:]}"

    degr_now = forced.get("degr", float(rng.beta(1.3, 7)))
    rated = float(spec["rated_life"])
    life_pct = forced.get("life", float(np.clip(rng.normal(50, 22), 3, 98)))
    op_hours = round(rated * life_pct / 100, 0)
    hsm_now = float(forced.get("hsm", float(np.clip(rng.gamma(2.0, 120), 5, 900))))
    age = round(float(min(2026 - year, rng.uniform(1, 2026 - year + 0.5))), 1)
    install = today - timedelta(days=int(op_hours / max(util, 0.1)))

    # sensor history: degradation drifts towards today's value
    drift = 0.35 if forced else float(rng.uniform(0, 0.05))
    days = np.arange(HISTORY_DAYS, 0, -1) - 1  # 29 .. 0 days ago
    path = np.clip(degr_now - drift * days / HISTORY_DAYS, 0, 1)
    temp, vib, pres = simulate_sensors(path, rng)
    for i, ago in enumerate(days):
        day = today - timedelta(days=int(ago))
        readings.append((comp_id, _dt(day, 6), round(op_hours - util * ago, 1),
                         round(max(hsm_now - util * ago, 0), 1), round(float(temp[i]), 1),
                         round(float(vib[i]), 2), round(float(pres[i]), 1), "telemetry"))

    # maintenance history consistent with hours since last maintenance
    n_maint = int(rng.integers(1, 6))
    last_day = today - timedelta(days=int(hsm_now / util))
    for k in range(n_maint):
        day = last_day - timedelta(days=int(k * rng.uniform(90, 200)))
        if day < install:
            break
        rtype = "Scheduled" if k == 0 else str(rng.choice(["Scheduled", "Inspection", "Unscheduled"]))
        records.append((comp_id, tail, _d(day), rtype, MAINT_DESCRIPTIONS[rtype],
                        round(max(op_hours - hsm_now - k * 150, 0), 0), None, None))

    # fault history in the last 24 months
    n_faults = int(forced.get("faults", rng.poisson(0.4 + 2 * degr_now)))
    for k in range(n_faults):
        key = str(rng.choice(list(FAULT_CODES)))
        code, desc = FAULT_CODES[key]
        recent = forced and k == 0
        day = today - timedelta(days=int(rng.integers(2, 12) if recent else rng.integers(20, 700)))
        severity = "High" if recent else str(rng.choice(["Low", "Medium"]))
        faults.append((f"F-{len(faults) + 1:04d}", tail, comp_id, _d(day), code,
                       f"{desc} ({ctype} {pos})", severity, "Open" if recent else "Closed"))

    return (comp_id, tail, ctype, pos, f"SN-{rng.integers(100000, 999999)}", _d(install),
            op_hours, rated, round(hsm_now, 1), age, n_faults, n_maint,
            round(float(temp[-1]), 1), round(float(vib[-1]), 2), round(float(pres[-1]), 1),
            _dt(today, 6))


def _parts(rng):
    rows = []
    stock_override = {"PN-HYD-2001": 0, "PN-PRP-1020": 0, "PN-PRP-1101": 2, "PN-ELE-4001": 1}
    for ctype, spec in config.COMPONENT_TYPES.items():
        system = spec["system"]
        rows.append((spec["replace_part"], f"{ctype} replacement assembly", ctype, "Replacement",
                     stock_override.get(spec["replace_part"], int(rng.integers(2, 5))), 0, 1,
                     int(rng.integers(14, 40)), float(rng.integers(40, 400)) * 1000, f"{system} store"))
        rows.append((spec["service_part"], f"{ctype} inspection & service kit", ctype, "Service kit",
                     stock_override.get(spec["service_part"], int(rng.integers(5, 12))), 0, 3,
                     int(rng.integers(4, 10)), float(rng.integers(2, 15)) * 1000, f"{system} store"))
    rows.append(("PN-GEN-9001", "O-ring and seal assortment", "General", "Consumable", 40, 0, 15, 5,
                 800.0, "Central store"))
    rows.append(("PN-GEN-9002", "Hydraulic filter element", "General", "Consumable", 6, 0, 8, 7,
                 1500.0, "Central store"))
    return rows


def _schedule(rng, aircraft_rows):
    rows = []
    for tail, *_, fh in aircraft_rows:
        for code, task, interval in (("PI", "Periodic inspection", 200), ("PH", "Phase inspection", 600)):
            since = float(rng.uniform(0, interval * 1.05))
            rows.append((f"{tail}-{code}", tail, f"{task} (fictional {interval} FH interval)",
                         float(interval), round(fh - since, 0)))
    return rows


def _purchase_orders(conn, today):
    conn.execute("INSERT INTO purchase_orders VALUES (?,?,?,?,?,?,?)",
                 ("PO-0001", "PN-HYD-2001", 2, _d(today - timedelta(days=12)),
                  _d(today + timedelta(days=6)), "Ordered", None))
    conn.execute("INSERT INTO purchase_orders VALUES (?,?,?,?,?,?,?)",
                 ("PO-0002", "PN-GEN-9001", 30, _d(today - timedelta(days=40)),
                  _d(today - timedelta(days=33)), "Received", _d(today - timedelta(days=34))))


def _work_orders(conn, rng, today, engine):
    """Seed active and completed work orders, then derive reservations from them."""
    from .risk import score_components  # local import avoids a cycle at module load

    comps = {r["component_id"]: dict(r) for r in conn.execute("SELECT * FROM components")}
    scores = {}
    if engine is not None:
        import pandas as pd
        scored = score_components(engine, pd.DataFrame(list(comps.values())))
        scores = dict(zip(scored["component_id"], scored["risk_score"]))

    def wo(wo_id, comp_id, action, severity, status, team, start_offset, grounding, created_ago, notes="",
           source="Predictive"):
        c = comps[comp_id]
        spec = config.COMPONENT_TYPES[c["component_type"]]
        part = spec["replace_part"] if action == "Replace" else spec["service_part"]
        hours = spec["replace_hours"] if action == "Replace" else spec["service_hours"]
        start = end = None
        if start_offset is not None:
            s = today + timedelta(days=start_offset)
            start, end = _d(s), _d(s + timedelta(days=math.ceil(hours / 8)))
        reserved = 1 if status in ("Open", "Scheduled", "In Progress") else 0
        created = _dt(today - timedelta(days=created_ago), 10)
        conn.execute(
            "INSERT INTO work_orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (wo_id, c["tail_no"], comp_id, f"{action}: {c['component_type']} {c['position']}", action,
             severity, scores.get(comp_id), None, part, 1, reserved, status, team, start, end, hours,
             int(grounding), source, created, created, None, notes))
        db.log_activity(conn, "Work order", f"{wo_id} created for {comp_id} ({severity}) - status {status}",
                        c["tail_no"])

    wo("WO-0001", "KM-303-GTE1", "Replace", "Critical", "In Progress", "T-PRP-A", -1, True, 3,
       "Engine removed; replacement assembly issued from store.")
    wo("WO-0002", "FT-205-G1", "Inspect & Service", "High", "In Progress", "T-ELE", 0, False, 2)
    wo("WO-0003", "AC-402-HP1", "Replace", "Critical", "Awaiting Parts", None, None, True, 4,
       "Replacement pump on order (PO-0001).")
    wo("WO-0004", "HU-504-LGA1", "Inspect & Service", "High", "Scheduled", "T-HYD", 2, False, 1)

    # routine backlog: older, low/medium-severity servicing waiting to be planned
    busy = {"KM-303-GTE1", "FT-205-G1", "AC-402-HP1", "HU-504-LGA1"}
    candidates = sorted(c for c, v in comps.items()
                        if c not in busy and v["component_type"] != "Gas Turbine Engine"
                        and (v["tail_no"], v["component_type"], v["position"]) not in FORCED)
    picks = rng.choice(len(candidates), size=9, replace=False)
    for k, idx in enumerate(sorted(int(i) for i in picks)):
        wo(f"WO-{k + 5:04d}", candidates[idx], "Inspect & Service", str(rng.choice(["Low", "Medium"])), "Open",
           None, None, False, int(rng.integers(6, 20)), "Routine servicing from fictional maintenance programme.", source="Scheduled")

    # completed history (closed work orders + matching maintenance records)
    comp_ids = sorted(comps)
    for k in range(12):
        cid = comp_ids[int(rng.integers(0, len(comp_ids)))]
        c = comps[cid]
        spec = config.COMPONENT_TYPES[c["component_type"]]
        done = today - timedelta(days=int(rng.integers(5, 120)))
        wo_id = f"WO-{k + 14:04d}"
        team = "T-GEN"
        conn.execute(
            "INSERT INTO work_orders VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (wo_id, c["tail_no"], cid, f"Inspect & Service: {c['component_type']} {c['position']}",
             "Inspect & Service", "Medium", None, None, spec["service_part"], 1, 0, "Completed", team,
             _d(done - timedelta(days=1)), _d(done), spec["service_hours"], 0, "Scheduled",
             _dt(done - timedelta(days=3)), _dt(done), _dt(done, 16), ""))
        conn.execute(
            "INSERT INTO maintenance_records (component_id, tail_no, performed_on, record_type, "
            "description, hours_at, team_id, work_order_id) VALUES (?,?,?,?,?,?,?,?)",
            (cid, c["tail_no"], _d(done), "Scheduled", f"{wo_id} completed: inspection & service",
             None, team, wo_id))

    conn.execute("""
        UPDATE spare_parts SET qty_reserved = COALESCE((
            SELECT SUM(w.part_qty) FROM work_orders w
            WHERE w.part_number = spare_parts.part_number AND w.parts_reserved = 1
              AND w.status IN ('Open','Scheduled','In Progress')), 0)
    """)
