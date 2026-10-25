from pathlib import Path

from streamlit.testing.v1 import AppTest

from aeroready import auth, db, seed, workorders

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def test_passwords_are_hashed_and_verified(fresh_db):
    auth.ensure_users()
    row = db.query_one("SELECT * FROM users WHERE username = 'admin'")
    assert "admin@123" not in row["password_hash"] and len(row["salt"]) == 32
    assert auth.authenticate("admin", "admin@123")["role"] == "Administrator"
    assert auth.authenticate(" ADMIN ", "admin@123") is not None  # username is case/space-insensitive
    assert auth.authenticate("admin", "wrong") is None
    assert auth.authenticate("nobody", "admin@123") is None
    assert auth.authenticate("", "") is None


def test_users_survive_demo_reset(fresh_db, engine):
    auth.ensure_users()
    seed.seed(engine)
    assert auth.authenticate("planner", "planner@123") is not None


def test_only_admin_can_reset():
    assert auth.can_reset({"role": "Administrator"})
    assert not auth.can_reset({"role": "Maintenance Planner"})
    assert not auth.can_reset(None)


def login(at, username, password):
    at.text_input(key="login_user").input(username)
    at.text_input(key="login_pass").input(password)
    at.button[[b.label for b in at.button].index("Sign in")].click()
    return at.run()


def test_app_requires_login_then_shows_dashboard(fresh_db):
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert not any("Fleet availability" in m.value for m in at.markdown), "dashboard visible before login"
    login(at, "admin", "bad-password")
    assert any("Invalid username or password" in e.value for e in at.error)
    login(at, "admin", "admin@123")
    assert not at.exception
    assert at.session_state["user"]["role"] == "Administrator"
    assert any("Fleet availability" in m.value for m in at.markdown)
    assert "reset_confirm" in [b.key for b in at.button]


def test_planner_role_cannot_reset_and_reset_keeps_admin_signed_in(fresh_db):
    at = AppTest.from_file(APP, default_timeout=120).run()
    login(at, "planner", "planner@123")
    assert not at.exception
    assert "reset_confirm" not in [b.key for b in at.button]

    workorders.create_work_order("FT-201", "FT-201-HP1", "Inspect & Service", "Medium")
    at = AppTest.from_file(APP, default_timeout=120).run()
    login(at, "admin", "admin@123")
    at.button(key="reset_confirm").click().run()
    assert not at.exception
    assert at.session_state["user"]["username"] == "admin"
    assert db.query_one("SELECT COUNT(*) n FROM work_orders WHERE component_id = 'FT-201-HP1'")["n"] == 0


def test_lockout_after_repeated_failures(fresh_db):
    at = AppTest.from_file(APP, default_timeout=120).run()
    for _ in range(5):
        login(at, "admin", "nope")
    assert any("paused" in e.value for e in at.error)
    login(at, "admin", "admin@123")  # correct password is still refused while locked
    assert "user" not in at.session_state
