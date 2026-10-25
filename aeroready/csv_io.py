"""Validated CSV import and export.

Imports are all-or-nothing: if any row fails validation, nothing is written and a
list of row-level errors is returned (row numbers match the spreadsheet, header = 1).
"""
from __future__ import annotations

import io

import pandas as pd

from . import config, db
from .workorders import allocate_parts

MAX_ROWS = 20000

IMPORT_SPECS = {
    "spare_parts": {
        "label": "Spare parts (insert new / update existing by part number)",
        "required": {
            "part_number": "str", "description": "str", "component_type": "str", "part_role": "str",
            "qty_on_hand": "int", "reorder_threshold": "int", "lead_time_days": "int",
        },
        "optional": {"unit_cost": ("float", 0.0), "location": ("str", "Central store")},
    },
    "sensor_readings": {
        "label": "Sensor readings (appended; newest reading becomes the component's current state)",
        "required": {
            "component_id": "str", "recorded_at": "datetime", "temp_deviation": "float",
            "vibration": "float", "pressure_deviation": "float",
        },
        "optional": {"operating_hours": ("float", None), "hours_since_maintenance": ("float", None)},
    },
    "maintenance_records": {
        "label": "Maintenance records (appended)",
        "required": {"component_id": "str", "performed_on": "date", "record_type": "str", "description": "str"},
        "optional": {"hours_at": ("float", None), "team_id": ("str", None)},
    },
}

RECORD_TYPES = ["Scheduled", "Unscheduled", "Inspection"]
PART_ROLES = ["Replacement", "Service kit", "Consumable"]


def template(kind: str) -> bytes:
    spec = IMPORT_SPECS[kind]
    cols = list(spec["required"]) + list(spec["optional"])
    return pd.DataFrame(columns=cols).to_csv(index=False).encode()


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def read_upload(data: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(data), dtype=str, keep_default_na=False, skipinitialspace=True)
    except UnicodeDecodeError:
        raise ValueError("File is not UTF-8 encoded text. Save it as 'CSV UTF-8' and try again.")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty.")
    except pd.errors.ParserError as exc:
        raise ValueError(f"Could not parse CSV: {exc}")
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    return df.apply(lambda s: s.str.strip())


def _convert(series: pd.Series, kind: str):
    blank = series == ""
    if kind == "int":
        num = pd.to_numeric(series, errors="coerce")
        bad = num.isna() | (num % 1 != 0)
        return num, bad
    if kind == "float":
        num = pd.to_numeric(series, errors="coerce")
        return num, num.isna()
    if kind == "date":
        d = pd.to_datetime(series, errors="coerce", format="ISO8601")
        return d.dt.date.astype(str), d.isna()
    if kind == "datetime":
        d = pd.to_datetime(series, errors="coerce", format="ISO8601")
        return d.dt.strftime("%Y-%m-%d %H:%M:%S"), d.isna()
    return series, blank


def validate(kind: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if kind not in IMPORT_SPECS:
        raise ValueError(f"Unsupported import type: {kind}")
    spec = IMPORT_SPECS[kind]
    errors: list[str] = []
    if df.empty:
        return df, ["The file has a header but no data rows."]
    if len(df) > MAX_ROWS:
        return df, [f"Too many rows ({len(df)}); the limit is {MAX_ROWS}."]
    missing = [c for c in spec["required"] if c not in df.columns]
    if missing:
        return df, [f"Missing required column(s): {', '.join(missing)}. "
                    f"Expected: {', '.join(list(spec['required']) + list(spec['optional']))}."]

    out = pd.DataFrame(index=df.index)

    def add_errors(mask, msg):
        for i in df.index[mask.fillna(True)]:
            errors.append(f"Row {i + 2}: {msg}")

    for col, kind_ in spec["required"].items():
        blank = df[col] == ""
        add_errors(blank, f"'{col}' is required.")
        val, bad = _convert(df[col], kind_)
        add_errors(bad & ~blank, f"'{col}' is not a valid {kind_}.")
        out[col] = val
    for col, (kind_, default) in spec["optional"].items():
        if col not in df.columns:
            out[col] = default
            continue
        blank = df[col] == ""
        val, bad = _convert(df[col], kind_)
        add_errors(bad & ~blank, f"'{col}' is not a valid {kind_}.")
        out[col] = val.where(~blank, default)

    # domain rules
    if kind == "spare_parts":
        for col in ["qty_on_hand", "reorder_threshold"]:
            add_errors(out[col] < 0, f"'{col}' must be zero or positive.")
        add_errors(out["lead_time_days"] < 1, "'lead_time_days' must be at least 1.")
        add_errors(~out["part_role"].isin(PART_ROLES), f"'part_role' must be one of {PART_ROLES}.")
        add_errors(~out["component_type"].isin(list(config.COMPONENT_TYPES) + ["General"]),
                   "'component_type' must be a known component type or 'General'.")
        dup = out["part_number"].duplicated(keep=False) & (out["part_number"] != "")
        add_errors(dup, "duplicate 'part_number' in file.")
        reserved = db.query_df("SELECT part_number, qty_reserved FROM spare_parts")
        res = out["part_number"].map(dict(zip(reserved["part_number"], reserved["qty_reserved"]))).fillna(0)
        add_errors(out["qty_on_hand"] < res, "'qty_on_hand' is below the quantity already reserved for work orders.")
    else:
        known = set(db.query_df("SELECT component_id FROM components")["component_id"])
        add_errors(~out["component_id"].isin(known) & (out["component_id"] != ""), "unknown 'component_id'.")
        if kind == "sensor_readings":
            for col in config.SENSOR_FEATURES:
                lo, hi, _ = config.FEATURE_RANGES[col]
                add_errors((out[col] < lo) | (out[col] > hi), f"'{col}' must be between {lo} and {hi}.")
        if kind == "maintenance_records":
            add_errors(~out["record_type"].isin(RECORD_TYPES), f"'record_type' must be one of {RECORD_TYPES}.")
            teams = set(db.query_df("SELECT team_id FROM teams")["team_id"])
            tid = out["team_id"]
            add_errors(tid.notna() & ~tid.isin(teams), "unknown 'team_id'.")

    errors = sorted(set(errors), key=lambda e: int(e.split(":")[0].split()[1]))
    return out, errors


def import_csv(kind: str, data: bytes) -> dict:
    """Validate and import. Returns {'rows': n, 'errors': [...]} - nothing is written on error."""
    try:
        df = read_upload(data)
    except ValueError as exc:
        return {"rows": 0, "errors": [str(exc)]}
    clean, errors = validate(kind, df)
    if errors:
        return {"rows": 0, "errors": errors}

    with db.connect() as conn:
        if kind == "spare_parts":
            for r in clean.to_dict("records"):
                conn.execute(
                    "INSERT INTO spare_parts (part_number, description, component_type, part_role, qty_on_hand, "
                    "qty_reserved, reorder_threshold, lead_time_days, unit_cost, location) "
                    "VALUES (?,?,?,?,?,0,?,?,?,?) ON CONFLICT(part_number) DO UPDATE SET "
                    "description=excluded.description, component_type=excluded.component_type, "
                    "part_role=excluded.part_role, qty_on_hand=excluded.qty_on_hand, "
                    "reorder_threshold=excluded.reorder_threshold, lead_time_days=excluded.lead_time_days, "
                    "unit_cost=excluded.unit_cost, location=excluded.location",
                    (r["part_number"], r["description"], r["component_type"], r["part_role"],
                     int(r["qty_on_hand"]), int(r["reorder_threshold"]), int(r["lead_time_days"]),
                     float(r["unit_cost"]), r["location"]))
        elif kind == "sensor_readings":
            comps = {r["component_id"]: dict(r) for r in conn.execute("SELECT * FROM components")}
            for r in clean.sort_values("recorded_at").to_dict("records"):
                c = comps[r["component_id"]]
                op = r["operating_hours"] if pd.notna(r["operating_hours"]) else c["operating_hours"]
                hsm = (r["hours_since_maintenance"] if pd.notna(r["hours_since_maintenance"])
                       else c["hours_since_maintenance"])
                conn.execute(
                    "INSERT INTO sensor_readings (component_id, recorded_at, operating_hours, hours_since_maintenance, "
                    "temp_deviation, vibration, pressure_deviation, source) VALUES (?,?,?,?,?,?,?,'csv-import')",
                    (r["component_id"], r["recorded_at"], op, hsm, r["temp_deviation"], r["vibration"],
                     r["pressure_deviation"]))
                if r["recorded_at"] >= (c["last_reading_at"] or ""):
                    conn.execute(
                        "UPDATE components SET temp_deviation=?, vibration=?, pressure_deviation=?, operating_hours=?, "
                        "hours_since_maintenance=?, last_reading_at=? WHERE component_id=?",
                        (r["temp_deviation"], r["vibration"], r["pressure_deviation"], op, hsm, r["recorded_at"],
                         r["component_id"]))
                    c.update(last_reading_at=r["recorded_at"], operating_hours=op, hours_since_maintenance=hsm)
        else:
            tails = dict(conn.execute("SELECT component_id, tail_no FROM components").fetchall())
            for r in clean.to_dict("records"):
                conn.execute(
                    "INSERT INTO maintenance_records (component_id, tail_no, performed_on, record_type, description, "
                    "hours_at, team_id, work_order_id) VALUES (?,?,?,?,?,?,?,NULL)",
                    (r["component_id"], tails[r["component_id"]], r["performed_on"], r["record_type"],
                     r["description"], None if pd.isna(r["hours_at"]) else float(r["hours_at"]),
                     None if pd.isna(r["team_id"]) else r["team_id"]))
        db.log_activity(conn, "Data import", f"Imported {len(clean)} row(s) into {kind}")
    if kind == "spare_parts":
        allocate_parts()
    return {"rows": len(clean), "errors": []}
