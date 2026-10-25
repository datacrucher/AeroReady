"""Shared Streamlit UI: theme CSS, KPI cards, badges, Plotly template and cached loaders."""
from __future__ import annotations

import base64
import html
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

from . import config, db, model, risk, seed

# --------------------------------------------------------------------------- palette (light theme)
BG = "#f6f8fb"
SURFACE = "#ffffff"
TEXT = "#0f172a"
MUTED = "#64748b"
GRID = "rgba(100,116,139,0.16)"
ACCENT = "#0e7490"

# categorical series (validated for a white surface; max 4 per chart, always with legend/labels)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# reserved status colours (darker steps so badge text stays readable on white) - always with a label/icon
STATUS_COLORS = {
    "Serviceable": "#15803d", "Under Maintenance": "#b45309", "Unavailable": "#b91c1c",
    "Low": "#15803d", "Medium": "#a16207", "High": "#c2410c", "Critical": "#b91c1c",
    "Available": "#15803d", "Low Stock": "#a16207", "Out of Stock": "#b91c1c",
    "Open": "#0369a1", "Awaiting Parts": "#c2410c", "Scheduled": "#6d28d9", "In Progress": "#b45309",
    "Completed": "#15803d", "Cancelled": "#64748b", "Ordered": "#0369a1", "Received": "#15803d",
}
STATUS_ICONS = {
    "Serviceable": "●", "Under Maintenance": "◐", "Unavailable": "■",
    "Low": "▁", "Medium": "▃", "High": "▅", "Critical": "▇",
    "Available": "●", "Low Stock": "◐", "Out of Stock": "■",
}

pio.templates["aeroready"] = go.layout.Template(layout=go.Layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT, size=12, family="Inter, Segoe UI, sans-serif"),
    colorway=SERIES,
    xaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False, tickfont=dict(color=MUTED), automargin=True,
               title=dict(standoff=8)),
    yaxis=dict(gridcolor=GRID, linecolor=GRID, zeroline=False, tickfont=dict(color=MUTED), automargin=True,
               title=dict(standoff=8)),
    # legend below the plot so it never collides with the title on narrow charts
    legend=dict(orientation="h", yref="container", yanchor="bottom", y=0, x=0, xanchor="left",
                font=dict(color=MUTED)),
    hoverlabel=dict(bgcolor="#ffffff", bordercolor="#cbd5e1", font=dict(color=TEXT)),
    margin=dict(l=8, r=8, t=40, b=8),
    title=dict(font=dict(size=14, color=TEXT), x=0, xanchor="left", y=0.97, yanchor="top", yref="container"),
))
pio.templates.default = "aeroready"

CSS = """
<style>
:root { --surface:#ffffff; --surface-2:#f8fafc; --border:#e2e8f0; --text:#0f172a; --muted:#64748b;
        --accent:#0e7490; }
.stApp { background:#f6f8fb; }
.block-container { padding-top:1.4rem; padding-bottom:2rem; max-width:1400px; }
h1 { font-size:1.45rem !important; font-weight:650 !important; letter-spacing:-.01em; margin-bottom:.1rem !important; }
h2 { font-size:1.12rem !important; font-weight:600 !important; }
h3 { font-size:1rem !important; font-weight:600 !important; }
section[data-testid="stSidebar"] { background:#ffffff; border-right:1px solid var(--border); }
.ar-brand { display:flex; align-items:center; gap:.55rem; padding:.2rem 0 .6rem 0; }
.ar-brand .logo { width:40px; height:40px; object-fit:contain; flex-shrink:0; }
.ar-brand .name { font-weight:650; color:var(--text); line-height:1.1; }
.ar-brand .sub { font-size:.72rem; color:var(--muted); }
.ar-nav-section { color:var(--muted); font-size:.7rem; font-weight:600; text-transform:uppercase; letter-spacing:.06em;
                  margin:.7rem 0 .1rem .2rem; }
section[data-testid="stSidebar"] [data-testid="stPageLink"] a { padding:.15rem .45rem; border-radius:6px; }
.ar-sub { color:var(--muted); font-size:.86rem; margin:-.2rem 0 .9rem 0; }
.ar-kpi { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:.7rem .85rem;
          height:100%; box-shadow:0 1px 2px rgba(15,23,42,.04); }
.ar-kpi .label { color:var(--muted); font-size:.74rem; text-transform:uppercase; letter-spacing:.04em; }
.ar-kpi .value { color:var(--text); font-size:1.55rem; font-weight:650; line-height:1.25; font-variant-numeric:tabular-nums; }
.ar-kpi .note { color:var(--muted); font-size:.75rem; }
.ar-kpi.accent { border-left:3px solid var(--accent); }
.ar-kpi.good { border-left:3px solid #16a34a; } .ar-kpi.warn { border-left:3px solid #ca8a04; }
.ar-kpi.bad { border-left:3px solid #dc2626; } .ar-kpi.serious { border-left:3px solid #ea580c; }
.ar-badge { display:inline-flex; align-items:center; gap:.3rem; padding:.08rem .5rem; border-radius:999px;
            font-size:.75rem; font-weight:600; border:1px solid; white-space:nowrap; }
.ar-panel { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:.8rem .95rem;
            box-shadow:0 1px 2px rgba(15,23,42,.04); }
.ar-alert { border-left:3px solid; padding:.45rem .7rem; margin-bottom:.4rem; background:var(--surface);
            border-radius:6px; font-size:.85rem; border-top:1px solid var(--border); border-right:1px solid var(--border);
            border-bottom:1px solid var(--border); }
.ar-alert .meta { color:var(--muted); font-size:.74rem; }
.ar-tag { color:var(--muted); font-size:.72rem; border:1px solid var(--border); border-radius:4px; padding:0 .35rem;
          background:var(--surface-2); }
.ar-gauge { font-size:2.2rem; font-weight:700; font-variant-numeric:tabular-nums; line-height:1; }
.ar-disclaimer { color:#64748b; font-size:.72rem; line-height:1.35; }
div[data-testid="stMetric"] { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:.6rem .8rem; }
.stTabs [data-baseweb="tab-list"] { gap:.25rem; }
.stTabs [data-baseweb="tab"] { padding:.35rem .8rem; }
@media (max-width: 640px) {
  .block-container { padding-left:1rem; padding-right:1rem; }
  .ar-kpi .value { font-size:1.25rem; }
}
</style>
"""


ASSETS = Path(__file__).resolve().parent.parent / "assets"
LOGO_ICON = ASSETS / "logo_icon.png"
LOGO_FULL = ASSETS / "logo_full.png"


@st.cache_data
def image_data_uri(path: str) -> str:
    """Inline an image as a data URI (empty string if the file is missing)."""
    try:
        return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()
    except OSError:
        return ""


def brand_html() -> str:
    src = image_data_uri(str(LOGO_ICON))
    logo = f"<img class='logo' src='{src}' alt='AeroReady logo'>" if src else ""
    return (f"<div class='ar-brand'>{logo}<div><div class='name'>AeroReady AI</div>"
            "<div class='sub'>Maintenance intelligence · demo</div></div></div>")


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f"<div class='ar-sub'>{html.escape(subtitle)}</div>", unsafe_allow_html=True)


def badge(label: str) -> str:
    color = STATUS_COLORS.get(label, MUTED)
    icon = STATUS_ICONS.get(label, "")
    return (f"<span class='ar-badge' style='color:{color};border-color:{color}55;background:{color}12'>"
            f"{icon} {html.escape(str(label))}</span>")


def kpi(label: str, value, note: str = "", tone: str = "accent") -> str:
    return (f"<div class='ar-kpi {tone}'><div class='label'>{html.escape(label)}</div>"
            f"<div class='value'>{html.escape(str(value))}</div><div class='note'>{html.escape(note)}</div></div>")


def kpi_row(cards: list[str]) -> None:
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        col.markdown(card, unsafe_allow_html=True)


def alert(level: str, title: str, meta: str = "") -> str:
    color = STATUS_COLORS.get(level, ACCENT)
    return (f"<div class='ar-alert' style='border-color:{color}'>{badge(level)} "
            f"<b>{html.escape(title)}</b><div class='meta'>{html.escape(meta)}</div></div>")


def flash(kind: str, message: str) -> None:
    """Queue a message that survives st.rerun()."""
    st.session_state.setdefault("_flash", []).append((kind, message))


def show_flash() -> None:
    for kind, msg in st.session_state.pop("_flash", []):
        getattr(st, kind)(msg)


def empty_state(message: str) -> None:
    st.markdown(f"<div class='ar-panel' style='color:{MUTED};text-align:center'>{html.escape(message)}</div>",
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- data access
@st.cache_resource(show_spinner="Training the experimental risk model on synthetic data (first launch only)...")
def get_engine() -> model.RiskEngine:
    return model.load_or_train()


def bootstrap() -> model.RiskEngine:
    engine = get_engine()
    if seed.ensure_seeded(engine):
        st.toast("Fictional demonstration dataset generated.")
    return engine


def scored_components() -> pd.DataFrame:
    """Every component with current risk - computed live so all pages stay consistent."""
    return risk.score_components(get_engine(), db.table("components"))


def component_label(row) -> str:
    return f"{row['component_id']} - {row['component_type']} {row['position']}"


def risk_color_scale():
    return [[0, STATUS_COLORS["Low"]], [0.2, STATUS_COLORS["Low"]], [0.2, STATUS_COLORS["Medium"]],
            [0.45, STATUS_COLORS["Medium"]], [0.45, STATUS_COLORS["High"]], [0.7, STATUS_COLORS["High"]],
            [0.7, STATUS_COLORS["Critical"]], [1, STATUS_COLORS["Critical"]]]


def history_with_risk(component_ids: list[str] | None = None) -> pd.DataFrame:
    """Sensor history with model risk for each reading (component age/faults held at current values)."""
    sql = ("SELECT r.*, c.rated_life_hours, c.age_years, c.prior_faults, c.tail_no, c.component_type "
           "FROM sensor_readings r JOIN components c USING (component_id)")
    params: tuple = ()
    if component_ids:
        sql += f" WHERE r.component_id IN ({','.join('?' * len(component_ids))})"
        params = tuple(component_ids)
    hist = db.query_df(sql + " ORDER BY r.recorded_at", params)
    if hist.empty:
        return hist
    feats = risk.to_features(hist)
    hist["risk_score"] = (get_engine().predict(feats) * 100).round(1)
    hist["health_index"] = (100 - hist["risk_score"]).round(1)
    hist["recorded_at"] = pd.to_datetime(hist["recorded_at"])
    return hist


# --------------------------------------------------------------------------- shared forms
def work_order_panel(comp: dict, assessment: dict, key: str, source: str = "Predictive") -> None:
    """Create a work order for ``comp`` from a risk ``assessment``, with a live parts check."""
    from . import workorders as wo_mod  # local import keeps ui import-light

    active = db.query_one(
        "SELECT wo_id, status, severity FROM work_orders WHERE component_id = ? AND status IN "
        "('Open','Awaiting Parts','Scheduled','In Progress')", (comp["component_id"],))
    if active:
        st.info(f"{comp['component_id']} already has active work order **{active['wo_id']}** "
                f"({active['severity']}, {active['status']}). Manage it in the Maintenance Planner.")
        if st.button("Open Maintenance Planner", key=f"{key}_goto"):
            st.switch_page("views/planner.py")
        return

    default_action = assessment.get("action") or "Inspect & Service"
    c1, c2, c3 = st.columns(3)
    action = c1.selectbox("Action", config.WO_ACTIONS, index=config.WO_ACTIONS.index(default_action),
                          key=f"{key}_action")
    severity = c2.selectbox("Severity", config.RISK_LEVELS,
                            index=config.RISK_LEVELS.index(assessment.get("level", "Low")), key=f"{key}_sev")
    hours = c3.number_input("Est. labour hours", min_value=1.0, max_value=500.0, step=1.0,
                            value=wo_mod.default_hours(comp["component_type"], action), key=f"{key}_hrs_{action}")
    parts = db.query_df("SELECT part_number, description FROM spare_parts ORDER BY part_number")
    default_part = wo_mod.default_part(comp["component_type"], action)
    p_opts = parts["part_number"].tolist()
    c4, c5 = st.columns([3, 1])
    part = c4.selectbox("Required spare part", p_opts, index=p_opts.index(default_part) if default_part in p_opts else 0,
                        format_func=lambda p: f"{p} - {parts.set_index('part_number').at[p, 'description']}",
                        key=f"{key}_part_{action}")
    qty = c5.number_input("Qty", min_value=0, max_value=50, value=1, step=1, key=f"{key}_qty")
    grounding = st.checkbox("Ground aircraft until this work is complete", value=bool(assessment.get("grounding")),
                            key=f"{key}_ground")
    notes = st.text_input("Notes (optional)", key=f"{key}_notes", max_chars=300)

    if qty > 0:
        av = wo_mod.part_availability(part, int(qty))
        msg = f"**Parts check - {part}:** {av['qty_on_hand']} on hand, {av['qty_reserved']} reserved, " \
              f"**{av['free']} free**."
        if av["sufficient"]:
            st.success(msg + f" {qty} unit(s) will be reserved; the work order can be scheduled immediately.")
        else:
            eta = (f"Purchase order {av['open_po']['po_id']} expected {av['open_po']['expected_on']}."
                   if av["open_po"] else f"Nothing on order - lead time {av['lead_time_days']} days. "
                                         "Raise an order in Spare-Parts Intelligence.")
            st.warning(msg + f" Insufficient stock: the work order will be created as **Awaiting Parts**. {eta}")

    if st.button("Create work order", type="primary", key=f"{key}_create"):
        try:
            res = wo_mod.create_work_order(
                comp["tail_no"], comp["component_id"], action, severity, part_number=part, part_qty=int(qty),
                est_hours=hours, grounding=grounding, risk_score=assessment.get("score"),
                recommended_action=assessment.get("text"), source=source, notes=notes)
        except wo_mod.WorkOrderError as exc:
            st.error(str(exc))
        else:
            flash("success" if res["parts_reserved"] else "warning", res["message"])
            st.rerun()


def assessment_card(a: dict, engine_label: str) -> None:
    color = STATUS_COLORS[a["level"]]
    st.markdown(
        f"<div class='ar-panel'><div class='label' style='color:{MUTED};font-size:.74rem;text-transform:uppercase'>"
        f"Predicted risk (next 50 FH, synthetic)</div>"
        f"<div style='display:flex;align-items:baseline;gap:.6rem;margin:.25rem 0'>"
        f"<span class='ar-gauge' style='color:{color}'>{a['score']:.0f}</span><span style='color:{MUTED}'>/ 100</span>"
        f"{badge(a['level'])}</div>"
        f"<div style='color:{MUTED};font-size:.8rem'>Health index {a['health_index']:.0f} / 100 · {html.escape(engine_label)}</div>"
        f"<div style='margin-top:.55rem;font-size:.88rem'><b>Recommendation:</b> {html.escape(a['text'])}</div></div>",
        unsafe_allow_html=True)


def explanation_chart(expl: pd.DataFrame, height: int = 280) -> go.Figure:
    """Diverging bar: contribution of each feature (orange raises risk, blue lowers it)."""
    e = expl.sort_values("contribution")
    colors = [SERIES[1] if v > 0 else SERIES[0] for v in e["contribution"]]
    fig = go.Figure(go.Bar(
        x=e["contribution"] * 100, y=e["label"], orientation="h", marker_color=colors,
        customdata=list(zip(e["value"], e["reference"])),
        hovertemplate="%{y}<br>value %{customdata[0]:.1f} (healthy ref. %{customdata[1]:.1f})"
                      "<br>contribution %{x:+.1f} pts<extra></extra>"))
    fig.add_vline(x=0, line_color="#475569", line_width=1)
    fig.update_layout(title="Why this score - contribution in risk points", height=height,
                      xaxis_title="risk points (+ raises, - lowers)", yaxis_title=None, showlegend=False)
    return fig
