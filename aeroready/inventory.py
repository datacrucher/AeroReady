"""Spare-parts intelligence: stock status, demand, reorder suggestions and purchase orders.

Reorder heuristic (transparent, not an optimiser):
    free            = on_hand - reserved
    projected_free  = free + on_order - waiting_demand - predicted_demand
    if projected_free <= reorder_threshold:
        suggested   = 2 * reorder_threshold - projected_free     (order up to 2x threshold)
``waiting_demand`` is the quantity needed by "Awaiting Parts" work orders.
``predicted_demand`` is one unit per High/Critical-risk component that has no active
work order and whose recommended action uses this part.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from . import config, db
from .workorders import WorkOrderError, allocate_parts, default_part


def stock_status(free: int, threshold: int) -> str:
    if free <= 0:
        return "Out of Stock"
    if free <= threshold:
        return "Low Stock"
    return "Available"


def predicted_part_demand(scored: pd.DataFrame | None) -> pd.Series:
    if scored is None or scored.empty:
        return pd.Series(dtype=int)
    active = db.query_df(
        f"SELECT component_id FROM work_orders WHERE status IN ({','.join('?' * len(config.ACTIVE_WO_STATUSES))})",
        tuple(config.ACTIVE_WO_STATUSES))
    risky = scored[scored["risk_level"].isin(["High", "Critical"])
                   & ~scored["component_id"].isin(active["component_id"])]
    parts = [default_part(r.component_type, r.recommended_action or "Inspect & Service")
             for r in risky.itertuples()]
    return pd.Series(parts, dtype=object).value_counts()


def stock_table(scored: pd.DataFrame | None = None) -> pd.DataFrame:
    parts = db.table("spare_parts")
    waiting = db.query_df(
        "SELECT part_number, SUM(part_qty) AS waiting_demand, COUNT(*) AS tasks_waiting "
        "FROM work_orders WHERE status = 'Awaiting Parts' GROUP BY part_number")
    on_order = db.query_df(
        "SELECT part_number, SUM(qty) AS on_order, MIN(expected_on) AS next_delivery "
        "FROM purchase_orders WHERE status = 'Ordered' GROUP BY part_number")
    df = parts.merge(waiting, on="part_number", how="left").merge(on_order, on="part_number", how="left")
    for col in ["waiting_demand", "tasks_waiting", "on_order"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    pred = predicted_part_demand(scored)
    df["predicted_demand"] = df["part_number"].map(pred).fillna(0).astype(int)
    df["free"] = df["qty_on_hand"] - df["qty_reserved"]
    df["status"] = [stock_status(f, t) for f, t in zip(df["free"], df["reorder_threshold"])]
    df["projected_free"] = df["free"] + df["on_order"] - df["waiting_demand"] - df["predicted_demand"]
    df["suggested_order"] = [
        max(0, 2 * t - p) if p <= t else 0 for p, t in zip(df["projected_free"], df["reorder_threshold"])
    ]
    return df


def place_order(part_number: str, qty: int, today: date | None = None) -> dict:
    if qty is None or int(qty) <= 0:
        raise WorkOrderError("Order quantity must be a positive integer.")
    today = today or date.today()
    with db.connect() as conn:
        part = conn.execute("SELECT * FROM spare_parts WHERE part_number = ?", (part_number,)).fetchone()
        if part is None:
            raise WorkOrderError(f"Unknown part number '{part_number}'.")
        po_id = db.next_id(conn, "purchase_orders", "po_id", "PO-")
        expected = today + timedelta(days=int(part["lead_time_days"]))
        conn.execute("INSERT INTO purchase_orders VALUES (?,?,?,?,?,?,NULL)",
                     (po_id, part_number, int(qty), today.isoformat(), expected.isoformat(), "Ordered"))
        db.log_activity(conn, "Parts", f"{po_id}: ordered {int(qty)} x {part_number}, expected {expected}")
    return {"po_id": po_id, "expected_on": expected}


def receive_order(po_id: str) -> list[str]:
    """Mark a purchase order received (simulated delivery) and allocate stock to waiting work."""
    with db.connect() as conn:
        po = conn.execute("SELECT * FROM purchase_orders WHERE po_id = ?", (po_id,)).fetchone()
        if po is None:
            raise WorkOrderError(f"Purchase order {po_id} not found.")
        if po["status"] != "Ordered":
            raise WorkOrderError(f"{po_id} is already {po['status'].lower()}.")
        conn.execute("UPDATE purchase_orders SET status = 'Received', received_on = ? WHERE po_id = ?",
                     (date.today().isoformat(), po_id))
        conn.execute("UPDATE spare_parts SET qty_on_hand = qty_on_hand + ? WHERE part_number = ?",
                     (po["qty"], po["part_number"]))
        db.log_activity(conn, "Parts", f"{po_id}: received {po['qty']} x {po['part_number']}")
    return allocate_parts()


def adjust_stock(part_number: str, qty_on_hand: int, reorder_threshold: int, lead_time_days: int) -> list[str]:
    """Manual stock-take correction. On-hand may not drop below reserved quantity."""
    for name, v in (("On-hand quantity", qty_on_hand), ("Reorder threshold", reorder_threshold)):
        if v is None or int(v) < 0:
            raise WorkOrderError(f"{name} must be zero or positive.")
    if lead_time_days is None or int(lead_time_days) < 1:
        raise WorkOrderError("Lead time must be at least 1 day.")
    with db.connect() as conn:
        part = conn.execute("SELECT * FROM spare_parts WHERE part_number = ?", (part_number,)).fetchone()
        if part is None:
            raise WorkOrderError(f"Unknown part number '{part_number}'.")
        if int(qty_on_hand) < part["qty_reserved"]:
            raise WorkOrderError(
                f"{part['qty_reserved']} unit(s) are reserved for work orders; on-hand cannot be lower.")
        conn.execute("UPDATE spare_parts SET qty_on_hand = ?, reorder_threshold = ?, lead_time_days = ? "
                     "WHERE part_number = ?", (int(qty_on_hand), int(reorder_threshold), int(lead_time_days),
                                               part_number))
        db.log_activity(conn, "Parts", f"Stock adjusted for {part_number}: on-hand {int(qty_on_hand)}")
    return allocate_parts()


def parts_ready_date(part_number: str | None, qty: int, today: date | None = None) -> tuple[date, str]:
    """Earliest date the part could be in hand for a waiting task, with an explanation."""
    today = today or date.today()
    if not part_number or qty == 0:
        return today, "no parts required"
    part = db.query_one("SELECT * FROM spare_parts WHERE part_number = ?", (part_number,))
    if part is None:
        return today, "unknown part"
    if part["qty_on_hand"] - part["qty_reserved"] >= qty:
        return today, "in stock"
    po = db.query_one("SELECT expected_on FROM purchase_orders WHERE part_number = ? AND status = 'Ordered' "
                      "ORDER BY expected_on LIMIT 1", (part_number,))
    if po:
        return max(today, date.fromisoformat(po["expected_on"])), f"on order, expected {po['expected_on']}"
    eta = today + timedelta(days=int(part["lead_time_days"]))
    return eta, f"not ordered - assumes order today ({part['lead_time_days']}-day lead time)"
