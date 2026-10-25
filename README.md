# AeroReady AI — Predictive Maintenance & Fleet Availability

**SIH 2026 · PS 26249 · Ministry of Defence / Defence Services Staff College · Transportation & Logistics**

AeroReady AI is a working prototype of an integrated maintenance-intelligence platform. It connects the whole chain in one application and one database:

**aircraft health data → predictive risk → work orders → spare-parts coordination → fleet availability**

> **Demonstration prototype.** All aircraft, components, part numbers, teams and data are fictional and synthetic. The risk model is trained and evaluated on synthetic data only and is **not** validated for real-world airworthiness or maintenance decisions.

---

## Quick start

```bash
# 1. (optional) create a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate

# 2. install dependencies
python -m pip install -r requirements.txt

# 3. run the app (opens http://localhost:8501)
python -m streamlit run app.py

# 4. run the tests
python -m pytest -q
```

**Sign in** with a demo account:

| Username | Password | Role |
|---|---|---|
| `admin` | `admin@123` | Administrator (full access, can reset demo data) |
| `planner` | `planner@123` | Maintenance Planner (full operational access, cannot reset) |

Passwords are stored as salted PBKDF2-SHA256 hashes in the local SQLite database. After 5 failed attempts, sign-in pauses for 30 seconds. Accounts survive **Reset demo data**. This is a demonstration login, not production identity management.

On first launch the app trains the risk model (about 2 seconds) and generates the fictional dataset in `data/`. Later launches reuse both. To start again from scratch, use **Reset demo data** in the sidebar, or delete the `data/` folder. Setting the `AEROREADY_DATA_DIR` environment variable points the app at a different data folder.

No internet access, API keys or external datasets are needed.

---

## 5-minute demo script (end-to-end acceptance flow)

| # | Step | Where | What you see |
|---|------|-------|--------------|
| 0 | Sign in as `admin` | Login page | The sidebar shows the user and role, with a sign-out button. |
| 1 | Inspect fleet readiness | **Fleet Command Centre** | Availability 87.5% (21/24). Two aircraft are under maintenance and one is grounded (AC-402, hydraulic pump with the part out of stock). Also shown: alerts, trends and the readiness board. |
| 2 | Select an aircraft | Pick **KM-306** on the dashboard, then **Open digital twin** | Every component of KM-306 with its risk badge. The fuel pump is Critical. |
| 3 | Change sensor values | Digital Twin, simulation controls | Move vibration, temperature or pressure. Risk, category and recommendation recompute instantly. Optionally tick *live-style telemetry*. |
| 4 | Explanation | Same page, plus **Predictive Maintenance**, then *What-if* / *Model card* | Contribution chart ("why this score"), the main drivers in plain language, and the model's held-out metrics. |
| 5 | Create a work order | *Create maintenance work order* expander | A live parts check runs before you submit. |
| 6 | Spare-part availability | Same form, then **Spare-Parts Intelligence** | PN-PRP-1020 is out of stock, so the work order becomes **Awaiting Parts** and KM-306 becomes *Unavailable*. Place a purchase order, then *Mark as received*. The stock is auto-allocated and the work order becomes **Open**. |
| 7 | Schedule / prioritise | **Maintenance Planner**, then *Suggested schedule*, *Generate*, *Apply* | Priority queue, Gantt chart by team, a "why this date" note for each job, and a no-double-booking check. Under *Manage*: start the work, then complete it. |
| 8 | Fleet impact | **Availability Simulator** and the dashboard | Scenario comparison (current, proactive, reactive, parts expedited, extra team, FIFO). After completion the aircraft returns to *Serviceable*. |
| 9 | Persistence | Restart `streamlit run app.py` | All work orders, stock, readings and logs are still there (SQLite). |

---

## What each module does

| Module | Page | Highlights |
|---|---|---|
| A. Fleet Command Centre | `views/dashboard.py` | KPIs computed from the database: availability, serviceable/under-maintenance/unavailable, components needing attention, high-priority and overdue work orders, critical spares shortages. Also a readiness board, live alerts, 30-day risk trend, weekly maintenance activity, risk by component type and an activity log. |
| B. Maintenance Data Hub | `views/data_hub.py` | Linked registry of aircraft, components, sensor readings, maintenance records, faults, work orders, purchase orders and the maintenance programme (flight-hour based). Search and filter on every table, an aircraft detail view, CSV export of any table, and **validated, all-or-nothing CSV import** (spare parts, sensor readings, maintenance records) with row-level errors. |
| C. Predictive Maintenance | `views/predictive.py` | Fleet risk ranking, per-component explanation, recommendation and work-order creation (single or bulk). An interactive what-if over all 7 features, and a full model card. |
| D. Component Digital Twin | `views/digital_twin.py` | Identity and usage, simulation controls, optional live-style noisy telemetry, health history, per-sensor trends, a linear-trend degradation projection ("days until High risk"), commit-reading-to-record, maintenance/fault history and work-order creation. |
| E. Maintenance Planner | `views/planner.py` | Priority queue with due dates and parts checks, a suggested schedule (heuristic, Gantt), manual assignment with overlap and skill validation, and the start / complete / unschedule / cancel lifecycle. |
| F. Spare-Parts Intelligence | `views/inventory.py` | Stock status, reserved/free stock, tasks waiting for parts, predicted demand from risky components, suggested reorder quantities, purchase orders, simulated deliveries with automatic allocation, stock-take adjustment, and shortage impact in aircraft-days. |
| G. Availability Simulator | `views/simulator.py` | Availability now, aircraft awaiting maintenance, planned downtime, before/after plan completion, up to 4 scenario curves, a per-aircraft downtime timeline, stated assumptions, and an automatically derived "binding constraint" reading. |

---

## How it works

### Architecture

```
app.py                     Streamlit entry point (login gate, navigation, sidebar, reset)
aeroready/
  config.py                paths, fictional catalogue, risk bands, statuses
  db.py                    SQLite schema and helpers (new connection per operation)
  seed.py                  reproducible synthetic fleet (24 aircraft, 150 components, 30-day telemetry)
  model.py                 synthetic training data, monotonic GBM, metrics, explanations, rule fallback
  risk.py                  feature derivation, fleet scoring, recommendations
  workorders.py            work-order lifecycle with consistent parts reservation
  inventory.py             stock status, reorder heuristic, purchase orders
  planner.py               greedy scheduling heuristic (no team overlap)
  availability.py          aircraft status rule, day-by-day simulator, scenarios
  csv_io.py                validated CSV import/export
  auth.py                  demo login: hashed passwords, roles
  ui.py                    theme, KPI cards, badges, Plotly template, shared forms
views/                     one file per module (A–G) + login.py
assets/login_bg.jpg        login-page background photo (user-supplied)
tests/                     pytest suite (model, workflow, planner, availability, CSV, UI smoke)
```

### Risk model (experimental)

* **Features:** operating hours as % of rated life, hours since last maintenance, component age, temperature deviation, vibration, pressure deviation, prior faults.
* **Synthetic data:** 8,000 records (seed 42). A hidden degradation variable drives noisy sensors and the probability of a synthetic "unscheduled removal within 50 FH". The model never sees the hidden variable.
* **Model:** `HistGradientBoostingClassifier` with **monotonic constraints**, so a worse input can never lower the predicted risk.
* **Evaluation:** stratified 75/25 split made *before* fitting, with fixed hyper-parameters. The app reports test ROC-AUC, average precision, Brier score, precision/recall at each risk band, the confusion matrix and the ROC curve, next to training ROC-AUC and a transparent rule-based baseline. Current run: test ROC-AUC ≈ 0.86, train ≈ 0.87, rule-based ≈ 0.86, so the ML model is *comparable* to the transparent rule score on this synthetic data. The app states this openly.
* **Explanations:** permutation importance (global). For a single prediction, each feature is reset to the healthy-component median and the change in risk is reported as that feature's contribution.
* **Fallback:** if the held-out ROC-AUC falls below 0.70, the app automatically switches to the weighted rule-based score.
* **Bands:** Low < 0.20 ≤ Medium < 0.45 ≤ High < 0.70 ≤ Critical.

### Business rules

* **Aircraft status:** *Under Maintenance* if any work order is In Progress. Otherwise *Unavailable* if any active work order grounds the aircraft. Otherwise *Serviceable*. **Availability = serviceable / total**, and this one rule is used everywhere.
* **Parts:** creating a work order reserves stock if enough is free; otherwise the order becomes *Awaiting Parts*. Deliveries and stock corrections auto-allocate to waiting orders by priority. Completion consumes the stock. Cancellation releases it. The invariant *reserved = Σ reserved active orders* is tested.
* **Completion:** writes a maintenance record, closes the component's open faults and restores its simulated condition (a replacement resets hours, age and faults; servicing removes most of the sensor deviation).
* **Scheduling heuristic:** jobs are ordered by grounding, then severity, then risk, then age. A job cannot start before its parts arrive (PO date, or the lead time if nothing is ordered). Each job goes to the qualified team that can **finish earliest**, and a team is never double-booked. This is a greedy rule of thumb and **not** a globally optimal scheduler.
* **Simulator:** an aircraft is unavailable during its maintenance windows, and also from the grounding date until the work ends. Scenario assumptions are shown in the app. The reactive scenario's failure timings are user-set and illustrative.

---

## Testing

```bash
python -m pytest -q
```

There are 48 tests:

* **Model:** reproducibility, held-out split, monotonicity, risk bands, rule fallback, sensor sensitivity, missing values.
* **Workflow:** creation, parts reservation, Awaiting Parts, the full lifecycle through to fleet availability, cancellation, persistence, telemetry validation, the reorder formula.
* **Planner and availability:** no team overlap across 20 random workloads, priority vs FIFO ordering, release dates, skills, the status rules, daily availability and scenarios.
* **CSV:** valid imports, row-level errors, all-or-nothing behaviour, the reserved-stock guard.
* **Auth:** password hashing, wrong/unknown credentials, login gate, role-based reset, lockout, accounts surviving a reset.
* **UI smoke:** every page renders headlessly; twin slider, then risk change, then work-order creation; planner generate and apply.

---

## Limitations (honest summary)

* All data is synthetic. Model metrics measure recovery of the synthetic generator's rules, **not** real-world predictive power. Real deployment would need real health-monitoring and maintenance data, independent validation and airworthiness approval.
* The digital twin is a software simulation (statistical sensor model plus linear-trend projection), not a physics or engineering model.
* Scheduling is a transparent greedy heuristic with whole-day granularity. It does not model shifts, hangar slots, tooling, technician certification or multi-part kits.
* Each work order uses a single part number. Purchase-order lead times are deterministic.
* Demo login only: two fixed roles, with no account management UI, password reset or concurrent-edit handling. Run it locally, not on a public network.
