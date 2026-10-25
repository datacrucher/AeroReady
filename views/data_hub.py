"""Module B - Unified Maintenance Data Hub."""
import pandas as pd
import streamlit as st

from aeroready import availability, csv_io, db, ui

ui.page_header("Maintenance Data Hub",
               "One connected data layer: aircraft registry, components, telemetry, maintenance history, faults, "
               "work orders, spares and schedules.")
ui.show_flash()

aircraft = db.table("aircraft")
wos = db.table("work_orders")
scored = ui.scored_components()
status = availability.aircraft_status(aircraft, wos)


def schedule_table() -> pd.DataFrame:
    s = db.query_df("SELECT s.*, a.total_flight_hours FROM maintenance_schedule s JOIN aircraft a USING (tail_no)")
    s["due_at_fh"] = s["last_done_fh"] + s["interval_fh"]
    s["remaining_fh"] = (s["due_at_fh"] - s["total_flight_hours"]).round(0)
    s["state"] = ["Overdue" if r < 0 else "Due soon" if r <= 0.1 * i else "OK"
                  for r, i in zip(s["remaining_fh"], s["interval_fh"])]
    return s


def table_view(df: pd.DataFrame, key: str, filters: list[str], column_config: dict | None = None,
               height: int = 420) -> None:
    cols = st.columns([3] + [2] * len(filters))
    q = cols[0].text_input("Search", key=f"{key}_q", placeholder="Search any column...")
    v = df
    for c, f in zip(cols[1:], filters):
        opts = sorted(v[f].dropna().astype(str).unique())
        sel = c.multiselect(f.replace("_", " ").capitalize(), opts, key=f"{key}_{f}")
        if sel:
            v = v[v[f].astype(str).isin(sel)]
    if q:
        mask = v.astype(str).apply(lambda col: col.str.contains(q, case=False, regex=False)).any(axis=1)
        v = v[mask]
    if v.empty:
        ui.empty_state("No records match.")
    else:
        st.dataframe(v, hide_index=True, height=height, column_config=column_config)
    c1, c2 = st.columns([3, 1])
    c1.caption(f"{len(v)} of {len(df)} records")
    c2.download_button("Export CSV", csv_io.to_csv_bytes(v), f"aeroready_{key}.csv", "text/csv", key=f"{key}_dl",
                       width="stretch")


tabs = st.tabs(["Aircraft detail", "Aircraft", "Components", "Sensor readings", "Maintenance records", "Faults",
                "Work orders", "Schedules", "Import / export"])

with tabs[0]:
    tails = aircraft["tail_no"].tolist()
    tail = st.selectbox("Aircraft", tails, index=tails.index(st.session_state["sel_tail"])
                        if st.session_state.get("sel_tail") in tails else 0, key="hub_tail")
    s = status.set_index("tail_no").loc[tail]
    st.markdown(f"### {tail} {ui.badge(s['status'])}", unsafe_allow_html=True)
    st.caption(f"{s['aircraft_type']} · {s['base']} · {s['squadron']} · built {s['year_built']} · "
               f"{s['total_flight_hours']:,.0f} flight hours" + (f" · {s['status_reason']}" if s["status_reason"] else ""))
    comps = scored[scored["tail_no"] == tail].sort_values("risk_score", ascending=False)
    st.dataframe(comps[["component_id", "component_type", "position", "serial_no", "operating_hours", "life_used_pct",
                        "hours_since_maintenance", "temp_deviation", "vibration", "pressure_deviation", "risk_score",
                        "risk_level", "recommendation"]],
                 hide_index=True, column_config={
                     "risk_score": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%.0f"),
                     "life_used_pct": st.column_config.NumberColumn("Life used %", format="%.0f")})
    b1, b2, _ = st.columns([1, 1, 3])
    comp_pick = b1.selectbox("Component", comps["component_id"], key="hub_comp", label_visibility="collapsed")
    if b2.button("Open in digital twin", width="stretch"):
        st.session_state["sel_tail"], st.session_state["sel_comp"] = tail, comp_pick
        st.switch_page("views/digital_twin.py")
    d1, d2 = st.columns(2, gap="medium")
    with d1:
        st.markdown("##### Work orders")
        w = wos[wos["tail_no"] == tail][["wo_id", "component_id", "action", "severity", "status", "scheduled_start",
                                         "scheduled_end"]]
        if w.empty:
            ui.empty_state("No work orders.")
        else:
            st.dataframe(w, hide_index=True)
        st.markdown("##### Scheduled maintenance programme")
        sch = schedule_table()
        st.dataframe(sch[sch["tail_no"] == tail][["task", "interval_fh", "last_done_fh", "due_at_fh", "remaining_fh",
                                                  "state"]], hide_index=True)
    with d2:
        st.markdown("##### Faults")
        f = db.query_df("SELECT fault_id, component_id, reported_on, fault_code, severity, status FROM faults "
                        "WHERE tail_no = ? ORDER BY reported_on DESC", (tail,))
        if f.empty:
            ui.empty_state("No faults recorded.")
        else:
            st.dataframe(f, hide_index=True)
        st.markdown("##### Maintenance records")
        mr = db.query_df("SELECT performed_on, component_id, record_type, description FROM maintenance_records "
                         "WHERE tail_no = ? ORDER BY performed_on DESC", (tail,))
        st.dataframe(mr, hide_index=True, height=240)

with tabs[1]:
    table_view(status, "aircraft", ["aircraft_type", "base", "status"])
with tabs[2]:
    table_view(scored[["component_id", "tail_no", "component_type", "position", "serial_no", "install_date",
                       "operating_hours", "rated_life_hours", "life_used_pct", "hours_since_maintenance", "age_years",
                       "prior_faults", "maintenance_count", "temp_deviation", "vibration", "pressure_deviation",
                       "risk_score", "risk_level", "last_reading_at"]],
               "components", ["component_type", "risk_level"])
with tabs[3]:
    r = db.query_df("SELECT r.*, c.tail_no FROM sensor_readings r JOIN components c USING (component_id) "
                    "ORDER BY r.recorded_at DESC")
    table_view(r, "sensor_readings", ["tail_no", "source"])
with tabs[4]:
    table_view(db.query_df("SELECT * FROM maintenance_records ORDER BY performed_on DESC"), "maintenance_records",
               ["record_type", "tail_no"])
with tabs[5]:
    table_view(db.query_df("SELECT * FROM faults ORDER BY reported_on DESC"), "faults", ["status", "severity"])
with tabs[6]:
    table_view(wos.sort_values("created_at", ascending=False), "work_orders", ["status", "severity"])
with tabs[7]:
    table_view(schedule_table(), "schedules", ["state", "tail_no"])

with tabs[8]:
    st.markdown("##### Import CSV")
    kind = st.selectbox("Data type", list(csv_io.IMPORT_SPECS), format_func=lambda k: csv_io.IMPORT_SPECS[k]["label"])
    spec = csv_io.IMPORT_SPECS[kind]
    st.caption("Required columns: " + ", ".join(f"`{c}`" for c in spec["required"])
               + " · optional: " + ", ".join(f"`{c}`" for c in spec["optional"])
               + ". Imports are all-or-nothing: any invalid row rejects the whole file.")
    st.download_button("Download blank template", csv_io.template(kind), f"template_{kind}.csv", "text/csv")
    up = st.file_uploader("CSV file", type=["csv"], key=f"up_{kind}")
    if up is not None and st.button("Validate and import", type="primary"):
        result = csv_io.import_csv(kind, up.getvalue())
        if result["errors"]:
            st.error(f"Import rejected - {len(result['errors'])} problem(s) found. Nothing was written.")
            st.dataframe(pd.DataFrame({"Problem": result["errors"][:200]}), hide_index=True)
        else:
            ui.flash("success", f"Imported {result['rows']} row(s) into {kind}.")
            st.rerun()
    st.divider()
    st.markdown("##### Export full tables")
    cols = st.columns(4)
    for i, name in enumerate(db.TABLES):
        cols[i % 4].download_button(name.replace("_", " ").capitalize(), csv_io.to_csv_bytes(db.table(name)),
                                    f"aeroready_{name}.csv", "text/csv", key=f"exp_{name}", width="stretch")
