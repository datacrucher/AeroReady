"""Module F - Spare-Parts Intelligence."""
from datetime import date, timedelta

import plotly.graph_objects as go
import streamlit as st

from aeroready import availability, db, inventory, ui, workorders

ui.page_header("Spare-Parts Intelligence",
               "Fictional spares inventory linked to work orders: reservations, shortages, purchase orders "
               "and their effect on the maintenance schedule.")
ui.show_flash()

today = date.today()
scored = ui.scored_components()
stock = inventory.stock_table(scored)
waiting = db.query_df(
    "SELECT w.wo_id, w.tail_no, w.component_id, w.severity, w.grounding, w.part_number, w.part_qty, w.created_at "
    "FROM work_orders w WHERE w.status = 'Awaiting Parts' ORDER BY w.grounding DESC, w.created_at")
pos = db.query_df("SELECT * FROM purchase_orders ORDER BY ordered_on DESC")

ui.kpi_row([
    ui.kpi("Part numbers", len(stock)),
    ui.kpi("Out of stock", int((stock["status"] == "Out of Stock").sum()), "no free units", "bad"),
    ui.kpi("Low stock", int((stock["status"] == "Low Stock").sum()), "at/below reorder threshold", "warn"),
    ui.kpi("Tasks waiting for parts", len(waiting), f"{int(waiting['grounding'].sum()) if len(waiting) else 0} "
                                                     "grounding an aircraft", "serious"),
    ui.kpi("Suggested reorders", int((stock["suggested_order"] > 0).sum()),
           f"{int((pos['status'] == 'Ordered').sum())} purchase orders open"),
])
st.write("")

tab_stock, tab_impact, tab_orders = st.tabs(["Inventory", "Shortage impact", "Orders & adjustments"])

with tab_stock:
    f1, f2, f3 = st.columns([2, 2, 3])
    f_status = f1.multiselect("Stock status", ["Available", "Low Stock", "Out of Stock"])
    f_type = f2.multiselect("Component type", sorted(stock["component_type"].unique()))
    search = f3.text_input("Search part number / description")
    v = stock
    if f_status:
        v = v[v["status"].isin(f_status)]
    if f_type:
        v = v[v["component_type"].isin(f_type)]
    if search:
        s = search.lower()
        v = v[v["part_number"].str.lower().str.contains(s, regex=False)
              | v["description"].str.lower().str.contains(s, regex=False)]
    if v.empty:
        ui.empty_state("No parts match these filters.")
    else:
        st.dataframe(
            v[["part_number", "description", "part_role", "qty_on_hand", "qty_reserved", "free", "reorder_threshold",
               "status", "lead_time_days", "tasks_waiting", "on_order", "next_delivery", "predicted_demand",
               "suggested_order", "location"]],
            hide_index=True, height=420,
            column_config={
                "part_number": "Part no.", "description": "Description", "part_role": "Role",
                "qty_on_hand": "On hand", "qty_reserved": "Reserved", "free": "Free", "reorder_threshold": "Reorder at",
                "status": "Status", "lead_time_days": "Lead time (d)", "tasks_waiting": "Tasks waiting",
                "on_order": "On order", "next_delivery": "Next delivery", "predicted_demand": "Predicted demand",
                "suggested_order": "Suggested order", "location": "Location",
            })
        st.caption("Free = on hand - reserved for work orders. Predicted demand = High/Critical-risk components "
                   "without a work order whose recommended action uses this part. Suggested order = order up to "
                   "2x reorder threshold when projected free stock (free + on order - waiting - predicted) is at or "
                   "below the threshold.")
        vv = v.sort_values("projected_free")
        fig = go.Figure()
        fig.add_bar(x=vv["part_number"], y=vv["projected_free"], name="Projected free stock",
                    marker_color=[ui.STATUS_COLORS["Out of Stock"] if p <= 0 else ui.STATUS_COLORS["Low Stock"]
                                  if p <= t else ui.SERIES[0] for p, t in zip(vv["projected_free"], vv["reorder_threshold"])],
                    hovertemplate="%{x}: %{y} projected free<extra></extra>")
        fig.add_scatter(x=vv["part_number"], y=vv["reorder_threshold"], mode="markers", name="Reorder threshold",
                        marker=dict(symbol="line-ew-open", size=18, line=dict(width=2, color=ui.TEXT)),
                        hovertemplate="%{x}: threshold %{y}<extra></extra>")
        fig.update_layout(title="Projected free stock vs reorder threshold (red ≤ 0, amber ≤ threshold)", height=300)
        st.plotly_chart(fig, config={"displayModeBar": False})

with tab_impact:
    if waiting.empty:
        ui.empty_state("No maintenance tasks are waiting for parts.")
    else:
        rows = []
        for r in waiting.itertuples():
            eta, why = inventory.parts_ready_date(r.part_number, int(r.part_qty), today)
            rows.append({"wo_id": r.wo_id, "aircraft": r.tail_no, "component": r.component_id, "severity": r.severity,
                         "grounds_aircraft": bool(r.grounding), "part": r.part_number, "qty": r.part_qty,
                         "earliest_parts_date": eta, "delay_days": (eta - today).days, "basis": why})
        st.markdown("**Maintenance tasks waiting for parts**")
        st.dataframe(rows, hide_index=True, column_config={
            "grounds_aircraft": st.column_config.CheckboxColumn("Grounds a/c"),
            "delay_days": st.column_config.NumberColumn("Parts delay (days)")})
    st.markdown("**Impact of shortages on the simulated schedule (30-day horizon)**")
    with st.spinner("Simulating schedules..."):
        res = availability.run_scenarios(scored, horizon=30, keys=["current", "expedited"])
    cur, exp = res["current"]["metrics"], res["expedited"]["metrics"]
    lost = cur["aircraft_days_lost"] - exp["aircraft_days_lost"]
    ui.kpi_row([
        ui.kpi("Aircraft-days lost - current", cur["aircraft_days_lost"], f"mean availability {cur['mean_availability_pct']}%",
               "serious"),
        ui.kpi("Aircraft-days lost - parts expedited", exp["aircraft_days_lost"],
               f"mean availability {exp['mean_availability_pct']}%", "good"),
        ui.kpi("Attributable to shortages", lost, "aircraft-days in the next 30 days", "bad" if lost > 0 else "good"),
    ])
    st.caption("Both figures come from the same scheduling heuristic and fleet data; the only difference is "
               "whether Awaiting Parts work orders can start today. See the Availability Simulator for details.")

with tab_orders:
    o1, o2 = st.columns(2, gap="medium")
    with o1:
        st.markdown("##### Place a purchase order")
        part_ids = stock.sort_values("suggested_order", ascending=False)["part_number"].tolist()
        sugg = dict(zip(stock["part_number"], stock["suggested_order"]))
        p = st.selectbox("Part", part_ids, format_func=lambda x: f"{x} - suggested {int(sugg[x])}", key="po_part")
        row = stock.set_index("part_number").loc[p]
        qty = st.number_input("Quantity", min_value=1, max_value=500, value=max(1, int(row["suggested_order"])),
                              key=f"po_qty_{p}")
        st.caption(f"Lead time {row['lead_time_days']} days → expected {today + timedelta(days=int(row['lead_time_days']))}.")
        if st.button("Place order", type="primary"):
            try:
                r = inventory.place_order(p, int(qty))
            except workorders.WorkOrderError as exc:
                st.error(str(exc))
            else:
                ui.flash("success", f"{r['po_id']} placed for {int(qty)} × {p}; expected {r['expected_on']}. "
                                    "Waiting work orders will use this date for scheduling.")
                st.rerun()
    with o2:
        st.markdown("##### Receive a delivery (simulated)")
        open_pos = pos[pos["status"] == "Ordered"]
        if open_pos.empty:
            ui.empty_state("No open purchase orders.")
        else:
            po_labels = {r.po_id: f"{r.po_id} - {r.qty} × {r.part_number} (exp. {r.expected_on})"
                         for r in open_pos.itertuples()}
            po = st.selectbox("Open purchase order", list(po_labels), format_func=po_labels.get)
            if st.button("Mark as received"):
                try:
                    allocated = inventory.receive_order(po)
                except workorders.WorkOrderError as exc:
                    st.error(str(exc))
                else:
                    ui.flash("success", f"{po} received into stock."
                             + (f" Parts automatically allocated to {', '.join(allocated)} (now Open, ready to schedule)."
                                if allocated else ""))
                    st.rerun()
    st.divider()
    st.markdown("##### Stock-take adjustment")
    a1, a2, a3, a4, a5 = st.columns([2, 1, 1, 1, 1], vertical_alignment="bottom")
    ap = a1.selectbox("Part", stock["part_number"], key="adj_part")
    arow = stock.set_index("part_number").loc[ap]
    on_hand = a2.number_input("On hand", min_value=0, value=int(arow["qty_on_hand"]), key=f"adj_oh_{ap}")
    thr = a3.number_input("Reorder at", min_value=0, value=int(arow["reorder_threshold"]), key=f"adj_thr_{ap}")
    lead = a4.number_input("Lead time (d)", min_value=1, value=int(arow["lead_time_days"]), key=f"adj_lt_{ap}")
    if a5.button("Save", width="stretch"):
        try:
            allocated = inventory.adjust_stock(ap, int(on_hand), int(thr), int(lead))
        except workorders.WorkOrderError as exc:
            st.error(str(exc))
        else:
            ui.flash("success", f"{ap} updated." + (f" Allocated to {', '.join(allocated)}." if allocated else ""))
            st.rerun()
    st.markdown("##### Purchase orders")
    if pos.empty:
        ui.empty_state("No purchase orders.")
    else:
        st.dataframe(pos, hide_index=True)
