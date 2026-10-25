"""Headless UI smoke tests: every page renders without exceptions, and key interactions work."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from aeroready import db

ROOT = Path(__file__).resolve().parent.parent
PAGES = ["app.py", "views/dashboard.py", "views/predictive.py", "views/digital_twin.py", "views/planner.py",
         "views/inventory.py", "views/simulator.py", "views/data_hub.py"]


def run(page, **state):
    at = AppTest.from_file(str(ROOT / page), default_timeout=120)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(fresh_db, page):
    at = run(page)
    assert not at.exception, [e.value for e in at.exception]


def test_twin_slider_changes_risk_and_creates_work_order(fresh_db):
    at = run("views/digital_twin.py", sel_tail="FT-201", sel_comp="FT-201-HP1")

    def gauge():
        for m in at.markdown:
            if "ar-gauge" in m.value:
                return int(m.value.split("ar-gauge' style='color:")[1].split(">")[1].split("<")[0])

    low = gauge()
    at.slider(key="tw_FT-201-HP1_vibration").set_value(15.0).run()
    at.slider(key="tw_FT-201-HP1_temp_deviation").set_value(40.0).run()
    assert not at.exception
    assert gauge() > low
    at.button(key="tw_FT-201-HP1_create").click().run()
    assert not at.exception
    assert db.query_one("SELECT COUNT(*) n FROM work_orders WHERE component_id='FT-201-HP1'")["n"] == 1


def test_planner_generate_and_apply(fresh_db):
    at = run("views/planner.py")
    at.button[[b.label for b in at.button].index("Generate suggested schedule")].click().run()
    assert not at.exception
    at.button[[b.label for b in at.button].index("Apply suggested schedule")].click().run()
    assert not at.exception
    assert db.query_one("SELECT COUNT(*) n FROM work_orders WHERE status='Scheduled'")["n"] > 1
