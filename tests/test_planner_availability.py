from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from aeroready import availability, config, db, planner, risk, workorders

TODAY = date(2026, 1, 5)
TEAMS = pd.DataFrame([t for t in config.TEAMS], columns=["team_id", "name", "systems", "hours_per_day", "service_only"])


def job(i, severity="Medium", ctype="Hydraulic Pump", action="Inspect & Service", hours=8.0, release=TODAY,
        grounding=False, created="2026-01-01", risk_score=30.0, tail=None):
    return planner.Job(wo_id=f"J{i}", tail_no=tail or f"AC-{i}", component_id=f"C{i}", component_type=ctype,
                       action=action, severity=severity, risk_score=risk_score, grounding=grounding,
                       est_hours=hours, release=release, release_reason="test", created_at=created)


def test_earliest_slot():
    booked = [(TODAY, TODAY + timedelta(days=2)), (TODAY + timedelta(days=3), TODAY + timedelta(days=5))]
    assert planner.earliest_slot(booked, TODAY, 1) == TODAY + timedelta(days=2)
    assert planner.earliest_slot(booked, TODAY, 2) == TODAY + timedelta(days=5)
    assert planner.earliest_slot([], TODAY, 3) == TODAY


def test_random_workloads_never_double_book_a_team():
    rng = np.random.default_rng(0)
    types = list(config.COMPONENT_TYPES)
    for trial in range(20):
        jobs = [job(i, severity=str(rng.choice(config.RISK_LEVELS)), ctype=str(rng.choice(types)),
                    action=str(rng.choice(config.WO_ACTIONS)), hours=float(rng.integers(4, 40)),
                    release=TODAY + timedelta(days=int(rng.integers(0, 10))), grounding=bool(rng.random() < 0.3))
                for i in range(30)]
        plan = planner.build_schedule(jobs, TEAMS, TODAY)
        assert not planner.has_team_overlap(plan)
        assert plan["team_id"].notna().all()
        assert (plan["start"] >= plan["release"]).all(), "job scheduled before its parts are ready"


def test_priority_order_critical_first():
    jobs = [job(1, "Low", created="2026-01-01"), job(2, "Critical", grounding=True, created="2026-01-04"),
            job(3, "High", created="2026-01-02")]
    plan = planner.build_schedule(jobs, TEAMS[TEAMS["team_id"] == "T-HYD"], TODAY)
    order = plan.sort_values("start")["wo_id"].tolist()
    assert order == ["J2", "J3", "J1"]
    fifo = planner.build_schedule(jobs, TEAMS[TEAMS["team_id"] == "T-HYD"], TODAY, order="fifo")
    assert fifo.sort_values("start")["wo_id"].tolist() == ["J1", "J3", "J2"]


def test_service_only_team_not_given_replacements():
    only_general = TEAMS[TEAMS["team_id"] == "T-GEN"]
    plan = planner.build_schedule([job(1, action="Replace")], only_general, TODAY)
    assert plan.iloc[0]["team_id"] is None and plan.iloc[0]["reason"] == "no qualified team"


def test_manual_assignment_rejects_overlap(fresh_db):
    a = workorders.create_work_order("FT-201", "FT-201-HP1", "Inspect & Service", "Low")["wo_id"]
    b = workorders.create_work_order("FT-202", "FT-202-HP1", "Inspect & Service", "Low")["wo_id"]
    start = date.today() + timedelta(days=20)
    workorders.assign_and_schedule(a, "T-HYD", start)
    with pytest.raises(workorders.WorkOrderError):
        workorders.assign_and_schedule(b, "T-HYD", start)
    with pytest.raises(workorders.WorkOrderError):
        workorders.assign_and_schedule(b, "T-ELE", start)  # not qualified for hydraulics


def test_apply_schedule_persists(fresh_db):
    plan = planner.build_schedule(planner.load_jobs(), db.table("teams"), date.today())
    n = planner.apply_schedule(plan)
    assert n > 0
    sched = db.query_df("SELECT * FROM work_orders WHERE status = 'Scheduled'")
    assert sched["team_id"].notna().all() and sched["scheduled_start"].notna().all()


def test_aircraft_status_rules():
    ac = pd.DataFrame({"tail_no": ["A", "B", "C", "D"]})
    wo = pd.DataFrame({
        "wo_id": ["1", "2", "3", "4"], "tail_no": ["A", "B", "C", "D"],
        "status": ["In Progress", "Awaiting Parts", "Open", "Completed"], "grounding": [0, 1, 0, 1]})
    st = availability.aircraft_status(ac, wo).set_index("tail_no")["status"]
    assert st.to_dict() == {"A": "Under Maintenance", "B": "Unavailable", "C": "Serviceable", "D": "Serviceable"}
    k = availability.fleet_kpis(availability.aircraft_status(ac, wo))
    assert k["availability_pct"] == 50.0


def test_daily_availability():
    plan = pd.DataFrame([
        {"tail_no": "A", "start": TODAY + timedelta(days=1), "end": TODAY + timedelta(days=3), "grounding": False,
         "down_from": None},
        {"tail_no": "B", "start": TODAY + timedelta(days=2), "end": TODAY + timedelta(days=4), "grounding": True,
         "down_from": None},
    ])
    d = availability.daily_availability(plan, ["A", "B", "C"], TODAY, 5)
    # day0: B grounded; day1: A down, B grounded; day2: A & B; day3: B; day4: all up
    assert d["available"].tolist() == [2, 1, 1, 2, 3]
    assert availability.summarise(d, 3)["aircraft_days_lost"] == 6


def test_scenarios_run_and_parts_expediting_never_hurts(fresh_db, engine):
    scored = risk.score_components(engine, db.table("components"))
    res = availability.run_scenarios(scored, horizon=30)
    assert set(res) == set(availability.SCENARIOS)
    for v in res.values():
        assert len(v["daily"]) == 30
        assert not planner.has_team_overlap(v["plan"])
    assert res["expedited"]["metrics"]["aircraft_days_lost"] <= res["current"]["metrics"]["aircraft_days_lost"]
    assert res["extra_team"]["metrics"]["aircraft_days_lost"] <= res["current"]["metrics"]["aircraft_days_lost"]
