from aeroready import csv_io, db


def test_valid_sensor_import_updates_component(fresh_db):
    data = (b"component_id,recorded_at,temp_deviation,vibration,pressure_deviation\n"
            b"FT-201-HP1,2099-01-01 08:00:00,12.5,6.1,4.0\n")
    res = csv_io.import_csv("sensor_readings", data)
    assert res == {"rows": 1, "errors": []}
    c = db.query_one("SELECT vibration FROM components WHERE component_id = 'FT-201-HP1'")
    assert c["vibration"] == 6.1


def test_invalid_rows_reject_whole_file(fresh_db):
    before = len(db.table("sensor_readings"))
    data = (b"component_id,recorded_at,temp_deviation,vibration,pressure_deviation\n"
            b"FT-201-HP1,2026-01-01,12.5,6.1,4.0\n"
            b"NOPE-1,2026-01-01,1,1,1\n"
            b"FT-201-HP1,not-a-date,abc,99,1\n")
    res = csv_io.import_csv("sensor_readings", data)
    assert res["rows"] == 0
    text = " ".join(res["errors"])
    assert "Row 3" in text and "unknown 'component_id'" in text
    assert "Row 4" in text and "recorded_at" in text and "temp_deviation" in text and "vibration" in text
    assert len(db.table("sensor_readings")) == before


def test_missing_columns_and_empty_file(fresh_db):
    assert "Missing required column" in csv_io.import_csv("spare_parts", b"part_number\nX\n")["errors"][0]
    assert csv_io.import_csv("spare_parts", b"")["errors"]
    assert csv_io.import_csv("spare_parts", b"\xff\xfe\x00bad")["errors"]


def test_spare_parts_upsert_and_reserved_guard(fresh_db):
    ok = (b"part_number,description,component_type,part_role,qty_on_hand,reorder_threshold,lead_time_days\n"
          b"PN-NEW-1,Test part,General,Consumable,5,2,7\n"
          b"PN-PRP-1020,Fuel pump assembly,Fuel Pump,Replacement,3,1,20\n")
    assert csv_io.import_csv("spare_parts", ok)["errors"] == []
    assert db.query_one("SELECT qty_on_hand FROM spare_parts WHERE part_number='PN-PRP-1020'")["qty_on_hand"] == 3
    reserved = db.query_one("SELECT part_number, component_type FROM spare_parts WHERE qty_reserved > 0")
    bad = (b"part_number,description,component_type,part_role,qty_on_hand,reorder_threshold,lead_time_days\n"
           + f"{reserved['part_number']},x,{reserved['component_type']},Replacement,0,1,5\n".encode())
    assert any("reserved" in e for e in csv_io.import_csv("spare_parts", bad)["errors"])


def test_maintenance_record_import(fresh_db):
    data = (b"component_id,performed_on,record_type,description,team_id\n"
            b"FT-201-HP1,2026-02-01,Inspection,Borescope check,T-HYD\n")
    assert csv_io.import_csv("maintenance_records", data)["rows"] == 1
    bad = b"component_id,performed_on,record_type,description\nFT-201-HP1,2026-02-01,Magic,x\n"
    assert any("record_type" in e for e in csv_io.import_csv("maintenance_records", bad)["errors"])
