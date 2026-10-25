"""Central configuration: paths, fictional component catalogue, risk bands and statuses.

Everything in this module describes a FICTIONAL demonstration fleet. None of the
aircraft types, part numbers, thresholds or limits correspond to real equipment.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("AEROREADY_DATA_DIR", ROOT / "data"))
DB_PATH = DATA_DIR / "aeroready.db"
MODEL_PATH = DATA_DIR / "risk_model.pkl"

RANDOM_SEED = 42
MODEL_VERSION = 4

# --------------------------------------------------------------------------- features
# Features used by both the ML model and the transparent rule-based score.
FEATURES = [
    "life_used_pct",
    "hours_since_maintenance",
    "age_years",
    "temp_deviation",
    "vibration",
    "pressure_deviation",
    "prior_faults",
]

FEATURE_LABELS = {
    "life_used_pct": "Operating hours (% of rated life)",
    "hours_since_maintenance": "Hours since last maintenance",
    "age_years": "Component age (years)",
    "temp_deviation": "Temperature deviation (°C)",
    "vibration": "Vibration (mm/s RMS)",
    "pressure_deviation": "Pressure deviation (%)",
    "prior_faults": "Prior faults (24 months)",
}

# (min, max, step) - used for sliders and CSV validation
FEATURE_RANGES = {
    "life_used_pct": (0.0, 130.0, 1.0),
    "hours_since_maintenance": (0.0, 1500.0, 10.0),
    "age_years": (0.0, 35.0, 0.5),
    "temp_deviation": (-10.0, 60.0, 0.5),
    "vibration": (0.0, 20.0, 0.1),
    "pressure_deviation": (0.0, 40.0, 0.5),
    "prior_faults": (0, 12, 1),
}

# Sensor channels that the digital twin streams (subset of FEATURES)
SENSOR_FEATURES = ["temp_deviation", "vibration", "pressure_deviation"]

# --------------------------------------------------------------------------- risk
# Probability thresholds -> category. Checked top-down.
RISK_BANDS = [(0.70, "Critical"), (0.45, "High"), (0.20, "Medium"), (0.0, "Low")]
RISK_LEVELS = ["Low", "Medium", "High", "Critical"]
SEVERITY_RANK = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}

# Minimum held-out ROC-AUC before the ML model is trusted; below it the app
# falls back to the transparent rule-based score.
MIN_ACCEPTABLE_AUC = 0.70

# --------------------------------------------------------------------------- work orders
WO_STATUSES = ["Open", "Awaiting Parts", "Scheduled", "In Progress", "Completed", "Cancelled"]
ACTIVE_WO_STATUSES = ["Open", "Awaiting Parts", "Scheduled", "In Progress"]
WO_ACTIONS = ["Inspect & Service", "Replace"]

AIRCRAFT_STATUSES = ["Serviceable", "Under Maintenance", "Unavailable"]

# --------------------------------------------------------------------------- catalogue
# Fictional component types. Each maps to a maintenance system (team skill), a
# replacement part, a service kit and nominal task durations.
COMPONENT_TYPES = {
    "Gas Turbine Engine": {
        "system": "Propulsion", "rated_life": 4000, "replace_part": "PN-PRP-1001",
        "service_part": "PN-PRP-1101", "replace_hours": 40, "service_hours": 12,
    },
    "Fuel Pump": {
        "system": "Propulsion", "rated_life": 3000, "replace_part": "PN-PRP-1020",
        "service_part": "PN-PRP-1120", "replace_hours": 10, "service_hours": 4,
    },
    "Hydraulic Pump": {
        "system": "Hydraulics", "rated_life": 3500, "replace_part": "PN-HYD-2001",
        "service_part": "PN-HYD-2101", "replace_hours": 12, "service_hours": 5,
    },
    "Landing Gear Actuator": {
        "system": "Hydraulics", "rated_life": 5000, "replace_part": "PN-LGR-3001",
        "service_part": "PN-LGR-3101", "replace_hours": 16, "service_hours": 6,
    },
    "Generator": {
        "system": "Electrical", "rated_life": 3000, "replace_part": "PN-ELE-4001",
        "service_part": "PN-ELE-4101", "replace_hours": 8, "service_hours": 4,
    },
    "Auxiliary Power Unit": {
        "system": "Propulsion", "rated_life": 3500, "replace_part": "PN-PRP-1050",
        "service_part": "PN-PRP-1150", "replace_hours": 20, "service_hours": 8,
    },
}

AIRCRAFT_TYPES = {
    "Falcon T-2 (trainer)": {"prefix": "FT"},
    "Kestrel MR-4 (multirole)": {"prefix": "KM"},
    "Albatross C-9 (transport)": {"prefix": "AC"},
    "Heron UH-3 (utility)": {"prefix": "HU"},
}

BASES = ["Base Alpha", "Base Bravo", "Base Charlie"]

# (team_id, name, comma-separated systems, hours per day, service_only)
# A service_only team may take "Inspect & Service" tasks but not replacements.
TEAMS = [
    ("T-PRP-A", "Propulsion Team A", "Propulsion", 8, 0),
    ("T-PRP-B", "Propulsion Team B", "Propulsion", 8, 0),
    ("T-HYD", "Hydraulics & Landing Gear Team", "Hydraulics", 8, 0),
    ("T-ELE", "Electrical & Avionics Team", "Electrical", 8, 0),
    ("T-GEN", "General Line Maintenance Team", "Propulsion,Hydraulics,Electrical", 8, 1),
]

DISCLAIMER = (
    "Demonstration prototype. All aircraft, components, part numbers, teams and data are "
    "fictional and synthetic. Predictions come from a model trained on synthetic data and are "
    "not validated for real-world airworthiness or maintenance decisions."
)
