"""AeroReady AI - integrated aircraft maintenance intelligence (SIH 2026, PS 26249).

Run with:  streamlit run app.py
"""
import streamlit as st

from aeroready import auth, config, seed, ui

st.set_page_config(page_title="AeroReady AI", page_icon=str(ui.LOGO_ICON) if ui.LOGO_ICON.exists() else "✈",
                   layout="wide", initial_sidebar_state="expanded")
ui.inject_css()
engine = ui.bootstrap()
auth.ensure_users()

user = st.session_state.get("user")
if not user:
    st.navigation([st.Page("views/login.py", title="Sign in", icon=":material/lock:")], position="hidden").run()
    st.stop()

pages = {
    "Operations": [
        st.Page("views/dashboard.py", title="Fleet Command Centre", icon=":material/dashboard:", default=True),
        st.Page("views/predictive.py", title="Predictive Maintenance", icon=":material/insights:"),
        st.Page("views/digital_twin.py", title="Component Digital Twin", icon=":material/memory:"),
    ],
    "Planning": [
        st.Page("views/planner.py", title="Maintenance Planner", icon=":material/event_note:"),
        st.Page("views/inventory.py", title="Spare-Parts Intelligence", icon=":material/inventory_2:"),
        st.Page("views/simulator.py", title="Availability Simulator", icon=":material/flight_takeoff:"),
    ],
    "Data": [
        st.Page("views/data_hub.py", title="Maintenance Data Hub", icon=":material/database:"),
    ],
}

# The built-in menu always renders above sidebar content, so it is hidden and rebuilt
# with page links underneath the brand.
nav = st.navigation(pages, position="hidden")

with st.sidebar:
    st.markdown(ui.brand_html(), unsafe_allow_html=True)
    for section, section_pages in pages.items():
        st.markdown(f"<div class='ar-nav-section'>{section}</div>", unsafe_allow_html=True)
        for page in section_pages:
            st.page_link(page)
    st.divider()
    u1, u2 = st.columns([3, 2], vertical_alignment="center")
    u1.markdown(f"**{user['full_name']}**  \n<span class='ar-tag'>{user['role']}</span>", unsafe_allow_html=True)
    if u2.button("Sign out", width="stretch"):
        st.session_state.clear()
        ui.flash("info", "You have been signed out.")
        st.rerun()
    method = "ML model" if engine.method == "ml" else "Rule-based fallback"
    st.caption(f"Risk engine: **{method}** · held-out ROC-AUC {engine.metrics.get('roc_auc', 0):.2f} "
               "(synthetic test data)")
    if auth.can_reset(user):
        with st.popover("Reset demo data", width="stretch"):
            st.write("Delete all changes and regenerate the fictional dataset? User accounts are kept.")
            if st.button("Yes, reset everything", type="primary", key="reset_confirm"):
                seed.seed(engine)
                st.session_state.clear()
                st.session_state["user"] = user  # stay signed in
                ui.flash("success", "Demo data reset to the original fictional dataset.")
                st.rerun()
    st.markdown(f"<div class='ar-disclaimer'>{config.DISCLAIMER}</div>", unsafe_allow_html=True)

nav.run()
