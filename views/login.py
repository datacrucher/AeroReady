"""Sign-in page shown before any operational page."""
import base64
import time
from pathlib import Path

import streamlit as st

from aeroready import auth, config, ui

MAX_ATTEMPTS = 5
LOCKOUT_SECONDS = 30

BG_IMAGE = Path(__file__).resolve().parent.parent / "assets" / "login_bg.jpg"


@st.cache_data
def background_css(path: str) -> str:
    """Hangar photo behind a dark navy overlay (keeps text readable, hides upscaling softness)."""
    try:
        data = base64.b64encode(Path(path).read_bytes()).decode()
    except OSError:
        return ".stApp { background:#0f172a; }"  # image missing: plain dark background
    return (".stApp { background: linear-gradient(rgba(8,15,30,.72), rgba(8,15,30,.82)), "
            f"url('data:image/jpeg;base64,{data}') center / cover no-repeat fixed; }}")


st.markdown(f"""
<style>
{background_css(str(BG_IMAGE))}
section[data-testid="stSidebar"], [data-testid="collapsedControl"] {{ display:none; }}
[data-testid="stHeader"] {{ background:transparent; }}
.ar-login-head {{ text-align:center; margin:2.2rem 0 1.2rem 0; }}
.ar-login-head .tile {{ display:inline-block; background:rgba(255,255,255,.97); border-radius:14px;
    padding:.7rem 1.2rem; box-shadow:0 12px 32px rgba(0,0,0,.35); }}
.ar-login-head .tile img {{ width:190px; max-width:60vw; display:block; }}
.ar-login-head .fallback {{ font-size:1.5rem; font-weight:650; color:#ffffff; }}
.ar-login-head .sub {{ color:#cbd5e1; font-size:.88rem; margin-top:.7rem; }}
[data-testid="stForm"] {{ background:rgba(255,255,255,.97); border-radius:12px; border:none !important;
    box-shadow:0 12px 32px rgba(0,0,0,.35); padding:1.2rem 1.3rem; }}
[data-testid="stExpander"] details {{ background:rgba(255,255,255,.95); border-radius:10px; }}
[data-testid="stAlert"] {{ background:rgba(255,255,255,.96); border-radius:8px; }}
.ar-login-foot {{ color:#cbd5e1; font-size:.72rem; line-height:1.35; text-align:center; margin-top:1rem; }}
</style>
""", unsafe_allow_html=True)

_, mid, _ = st.columns([1, 1.25, 1])
with mid:
    logo = ui.image_data_uri(str(ui.LOGO_FULL))
    head = (f"<div class='tile'><img src='{logo}' alt='AeroReady - AI aircraft maintenance'></div>" if logo
            else "<div class='fallback'>AeroReady AI</div>")
    st.markdown(f"<div class='ar-login-head'>{head}"
                "<div class='sub'>Predictive maintenance &amp; fleet availability</div></div>",
                unsafe_allow_html=True)
    ui.show_flash()

    locked_until = st.session_state.get("login_locked_until", 0)
    remaining = int(locked_until - time.time())

    with st.form("login", border=True):
        st.markdown("**Sign in**")
        username = st.text_input("Username", key="login_user", autocomplete="username")
        password = st.text_input("Password", type="password", key="login_pass", autocomplete="current-password")
        submitted = st.form_submit_button("Sign in", type="primary", width="stretch")

    if submitted:
        if remaining > 0:
            st.error(f"Too many failed attempts. Try again in {remaining} s.")
        elif not username.strip() or not password:
            st.error("Enter both a username and a password.")
        else:
            user = auth.authenticate(username, password)
            if user:
                st.session_state.pop("login_failures", None)
                st.session_state["user"] = user
                st.rerun()
            failures = st.session_state.get("login_failures", 0) + 1
            st.session_state["login_failures"] = failures
            if failures >= MAX_ATTEMPTS:
                st.session_state["login_locked_until"] = time.time() + LOCKOUT_SECONDS
                st.session_state["login_failures"] = 0
                st.error(f"Too many failed attempts. Sign-in is paused for {LOCKOUT_SECONDS} s.")
            else:
                st.error("Invalid username or password.")

    with st.expander("Demo accounts"):
        st.markdown("\n".join(f"- **{u}** / `{p}` - {role}: {auth.ROLES[role]}"
                              for u, p, _, role in auth.DEMO_USERS))
    st.markdown(f"<div class='ar-login-foot'>{config.DISCLAIMER}</div>",
                unsafe_allow_html=True)
