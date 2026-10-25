from datetime import date

import pytest

from aeroready import availability, db, inventory, risk, workorders


def part(pn):
    return db.query_one("SELECT * FROM spare_parts WHERE part_number = ?", (pn,))


def reservation_invariant_holds() -> bool:
    rows = db.query_df("""
        SELECT p.part_number, p.qty_reserved,
               COALESCE((SELECT SUM(part_qty) FROM work_orders w WHERE w.part_number = p.part_number
                         AND w.parts_reserved = 1 AND w.status IN ('Open','Scheduled','In Progress')), 0) AS expected
        FROM spare_parts p""")
    return bool((rows["qty_reserved"] == rows["expected"]).all())


def status_of(tail):
    return availability.aircraft_status(db.table("aircraft"), db.table("work_orders")).set_index("tail_no").at[tail, "status"]


def kpis():
    return availability.fleet_kpis(availability.aircraft_status(db.table("aircraft"), db.table("work_orders")))


def test_seed_creates_consistent_dataset(fresh_db):
    assert len(db.table("aircraft")) == 24
    assert len(db.table("components")) > 100
    assert reservation_invariant_holds()
    k = kpis()
    assert k["total"] == k["serviceable"] + k["under_maintenance"] + k["unavailable"]
    assert k["availability_pct"] == round(100 * k["serviceable"] / k["total"], 1)


def test_create_reserves_parts_when_available(fresh_db):
    comp = db.query_one("SELECT * FROM components WHERE component_id = 'FT-201-HP1'")
    pn = workorders.default_part(comp["component_type"], "Inspect & Service")
    before = part(pn)
    res = workorders.create_work_order("FT-201", "FT-201-HP1", "Inspect & Service", "Medium")
    assert res["status"] == "Open" and res["parts_reserved"]
    assert part(pn)["qty_reserved"] == before["qty_reserved"] + 1
    assert reservation_invariant_holds()


def test_out_of_stock_creates_awaiting_parts_and_grounds(fresh_db):
    before = kpis()
    res = workorders.create_work_order("KM-306", "KM-306-FP1", "Replace", "Critical", grounding=True)
    assert res["status"] == "Awaiting Parts" and not res["parts_reserved"]
    after = kpis()
    assert after["unavailable"] == before["unavailable"] + 1
    assert after["availability_pct"] < before["availability_pct"]


def test_validation_errors(fresh_db):
    with pytest.raises(workorders.WorkOrderError):
        workorders.create_work_order("FT-201", "KM-306-FP1", "Replace", "High")  # wrong aircraft
    with pytest.raises(workorders.WorkOrderError):
        workorders.create_work_order("FT-201", "FT-201-HP1", "Overhaul", "High")  # bad action
    workorders.create_work_order("FT-201", "FT-201-HP1", "Inspect & Service", "Low")
    with pytest.raises(workorders.WorkOrderError):
        workorders.create_work_order("FT-201", "FT-201-HP1", "Inspect & Service", "Low")  # duplicate


def test_full_lifecycle_updates_stock_component_and_availability(fresh_db):
    wo = workorders.create_work_order("KM-306", "KM-306-FP1", "Replace", "Critical", grounding=True)["wo_id"]
    with pytest.raises(workorders.WorkOrderError):
        workorders.start_work(wo)  # parts not reserved
    po = inventory.place_order("PN-PRP-1020", 2)
    assert inventory.receive_order(po["po_id"]) == [wo]
    assert db.query_one("SELECT status FROM work_orders WHERE wo_id = ?", (wo,))["status"] == "Open"
    workorders.assign_and_schedule(wo, "T-PRP-B", date.today())
    workorders.start_work(wo)
    assert status_of("KM-306") == "Under Maintenance"
    on_hand = part("PN-PRP-1020")["qty_on_hand"]
    workorders.complete_work_order(wo, "replaced")
    assert part("PN-PRP-1020")["qty_on_hand"] == on_hand - 1
    comp = db.query_one("SELECT * FROM components WHERE component_id = 'KM-306-FP1'")
    assert comp["operating_hours"] == 0 and comp["hours_since_maintenance"] == 0
    assert db.query_one("SELECT COUNT(*) n FROM faults WHERE component_id='KM-306-FP1' AND status='Open'")["n"] == 0
    assert db.query_one("SELECT COUNT(*) n FROM maintenance_records WHERE work_order_id = ?", (wo,))["n"] == 1
    assert status_of("KM-306") == "Serviceable"
    assert reservation_invariant_holds()


def test_completion_lowers_component_risk(fresh_db, engine):
    before = risk.score_components(engine, db.table("components")).set_index("component_id").at["KM-303-GTE1", "risk_score"]
    workorders.complete_work_order("WO-0001")  # seeded in-progress engine replacement
    after = risk.score_components(engine, db.table("components")).set_index("component_id").at["KM-303-GTE1", "risk_score"]
    assert after < before


def test_cancel_releases_reservation(fresh_db):
    res = workorders.create_work_order("FT-201", "FT-201-HP1", "Inspect & Service", "Low")
    reserved = part(res["part_number"])["qty_reserved"]
    workorders.cancel_work_order(res["wo_id"], "test")
    assert part(res["part_number"])["qty_reserved"] == reserved - 1
    assert reservation_invariant_holds()


def test_changes_persist_across_connections(fresh_db):
    res = workorders.create_work_order("FT-201", "FT-201-HP1", "Inspect & Service", "Low")
    # every db call opens a new connection, mirroring an application restart
    assert db.query_one("SELECT status FROM work_orders WHERE wo_id = ?", (res["wo_id"],))["status"] == "Open"
    assert db.query_one("SELECT COUNT(*) n FROM activity_log WHERE message LIKE ?", (f"{res['wo_id']}%",))["n"] >= 1


def test_record_reading_validates_and_updates(fresh_db):
    with pytest.raises(workorders.WorkOrderError):
        workorders.record_reading("FT-201-HP1", 999, 2, 2)
    workorders.record_reading("FT-201-HP1", 20.0, 9.0, 12.0)
    c = db.query_one("SELECT * FROM components WHERE component_id = 'FT-201-HP1'")
    assert (c["temp_deviation"], c["vibration"], c["pressure_deviation"]) == (20.0, 9.0, 12.0)


def test_reorder_suggestion(fresh_db, engine):
    scored = risk.score_components(engine, db.table("components"))
    table = inventory.stock_table(scored).set_index("part_number")
    for _, r in table.iterrows():
        assert r["free"] == r["qty_on_hand"] - r["qty_reserved"]
        assert r["status"] == inventory.stock_status(r["free"], r["reorder_threshold"])
        if r["projected_free"] <= r["reorder_threshold"]:
            assert r["suggested_order"] == max(0, 2 * r["reorder_threshold"] - r["projected_free"])
        else:
            assert r["suggested_order"] == 0
    with pytest.raises(workorders.WorkOrderError):
        inventory.place_order("PN-PRP-1020", 0)


def test_stock_adjustment_cannot_drop_below_reserved(fresh_db):
    reserved_part = db.query_one("SELECT * FROM spare_parts WHERE qty_reserved > 0")
    with pytest.raises(workorders.WorkOrderError):
        inventory.adjust_stock(reserved_part["part_number"], reserved_part["qty_reserved"] - 1, 1, 5)
