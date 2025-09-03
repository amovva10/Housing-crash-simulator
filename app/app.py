import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import streamlit as st
import pandas as pd
import altair as alt

from src.scenario import baseline_and_scenario, load_model_and_metrics

st.set_page_config(page_title="Housing Crash Simulator", layout="centered")

def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

st.title("Housing Crash Simulator")
st.caption("Change a few settings → see estimated next month home prices and crash risk.")

if "has_run" not in st.session_state:
    st.session_state.has_run = False
if "active_inputs" not in st.session_state:
    # (interest, mortgage, unemployment)
    st.session_state.active_inputs = (0.0, 0.0, 0.0)  
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "mode" not in st.session_state:
    st.session_state.mode = "Normal"

mode = st.radio(
    "Choose a scenario",
    ["Normal", "Personal"],
    horizontal=True,
    index=["Normal", "Personal"].index(st.session_state.mode if st.session_state.mode in ["Normal","Personal"] else "Normal"),
)
st.session_state.mode = mode

interest_pp, mortgage_pp, unemp_pp = 0.0, 0.0, 0.0

if mode == "Personal":
    st.write("Set additive changes (percentage points). Example: +1.0 turns 5.0% → 6.0%.")

    # ensure slider keys exist once
    for k, default in [("interest_pp", 0.0), ("mortgage_pp", 0.0), ("unemp_pp", 0.0)]:
        if k not in st.session_state:
            st.session_state[k] = default

    #  buttons to push the dot into zones
    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    if bcol1.button("Upside (GREEN)"):
        st.session_state.interest_pp = -2.0
        st.session_state.mortgage_pp = -2.0
        st.session_state.unemp_pp    = -1.0
        safe_rerun()
    if bcol2.button("Low risk"):
        st.session_state.interest_pp = +0.5
        st.session_state.mortgage_pp = +0.5
        st.session_state.unemp_pp    = +0.2
        safe_rerun()
    if bcol3.button("Medium risk"):
        st.session_state.interest_pp = +1.5
        st.session_state.mortgage_pp = +1.5
        st.session_state.unemp_pp    = +0.8
        safe_rerun()
    if bcol4.button("High risk (RED)"):
        st.session_state.interest_pp = +3.0
        st.session_state.mortgage_pp = +3.0
        st.session_state.unemp_pp    = +2.5
        safe_rerun()

    interest_pp = st.slider("Interest rate change (pp)", -3.0, 3.0, st.session_state.interest_pp, 0.1, key="interest_pp")
    mortgage_pp = st.slider("Mortgage rate change (pp)", -3.0, 3.0, st.session_state.mortgage_pp, 0.1, key="mortgage_pp")
    unemp_pp    = st.slider("Unemployment rate change (pp)", -3.0, 3.0, st.session_state.unemp_pp, 0.1, key="unemp_pp")

pending_inputs = (float(interest_pp), float(mortgage_pp), float(unemp_pp))

left, right = st.columns([3, 1])
run   = left.button("Show result", type="primary", use_container_width=True)
reset = right.button("Reset", use_container_width=True)

if reset:
    st.session_state.has_run = False
    st.session_state.last_result = None
    st.session_state.active_inputs = (0.0, 0.0, 0.0)
    st.session_state.mode = "Normal"
    for k in ("interest_pp", "mortgage_pp", "unemp_pp"):
        if k in st.session_state:
            st.session_state[k] = 0.0
    safe_rerun()

if run:
    st.session_state.active_inputs = pending_inputs
    i_pp, m_pp, u_pp = st.session_state.active_inputs
    st.session_state.last_result = baseline_and_scenario(
        interest_pp=i_pp, mortgage_pp=m_pp, unemp_pp=u_pp
    )
    st.session_state.has_run = True

if not st.session_state.has_run:
    st.info("Pick a scenario and click **Show result** to see next-month prices and crash risk.")
    st.stop()

res = st.session_state.last_result
i_pp, m_pp, u_pp = st.session_state.active_inputs

latest  = res["last_actual"]
scn     = res["scenario_pred"]
pct_scn = res.get("pct_change_scenario", (scn - latest) / latest if latest else 0.0) * 100

# robust crash probability
crash_prob = res.get("crash_prob_scenario", None)
if crash_prob is None:
    crash_prob = max(0.0, min(1.0, - (pct_scn / 100.0) / 0.10))

# Choice label 
def choice_from_inputs(tup):
    if tup == (0.0, 0.0, 0.0): return "Normal"
    return "Personal"

choice_label = choice_from_inputs((i_pp, m_pp, u_pp))

risk = "Low"; color = "#10b981"   
if crash_prob >= 0.66:
    risk, color = "High", "#ef4444"   
elif crash_prob >= 0.33:
    risk, color = "Medium", "#f59e0b" 

st.markdown(
    f"""
<div style="border:1px solid rgba(0,0,0,.06); padding:14px 16px; border-radius:12px;">
  <div style="font-weight:600; margin-bottom:6px;">Crash risk</div>
  <div style="display:flex; align-items:center; gap:10px;">
    <span style="background:{color}; color:white; padding:4px 10px; border-radius:999px; font-size:0.9rem;">{risk}</span>
    <span style="opacity:.8;">(~{crash_prob*100:.0f}%)</span>
  </div>
  <div style="height:8px; background:#eee; border-radius:999px; margin-top:10px;">
    <div style="height:8px; width:{crash_prob*100:.0f}%; background:{color}; border-radius:999px;"></div>
  </div>
  <div style="margin-top:8px; opacity:.85;">
    Choice: <b>{choice_label}</b>. Under this scenario, next month’s prices are expected to
    {'rise' if pct_scn>0 else 'fall' if pct_scn<0 else 'stay about the same'} by about
    <b>{abs(pct_scn):.1f}%</b>.
  </div>
  <div style="margin-top:4px; font-size:.9rem; opacity:.7;">
    (Crash risk uses a simple rule: a 10% 1-month drop ≈ 100% risk. It’s a stress test, not actual financial advice.)
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Latest home price level**")
    st.markdown(f"<small>{latest:,.0f}</small>", unsafe_allow_html=True)
with c2:
    st.markdown("**Your choice (next month)**")
    st.markdown(f"<small>{scn:,.0f} · {pct_scn:+.1f}% vs latest</small>", unsafe_allow_html=True)

st.caption(
    "**HPI** = *Home Price Index*. Score for average home prices (not dollars). "
    "The exact number (e.g., 297) is an index level — what matters most is the **% change**."
)

st.divider()

pct = float(pct_scn)

bands = pd.DataFrame({
    "band":  ["High risk", "Medium risk", "Low risk", "Upside"],
    "x0":    [-10.0, -5.0, -2.0, 0.0],   
    "x1":    [ -5.0, -2.0,  0.0, 2.0],   
    "color": ["#fdecea", "#fff7e6", "#e8f5e9", "#e8f0fe"], 
})

xmin = min(-12.0, pct - 3.0)
xmax = max(  4.0, pct + 3.0)

bands_chart = alt.Chart(bands).mark_rect().encode(
    x=alt.X("x0:Q", title="Change vs latest (%)", scale=alt.Scale(domain=[xmin, xmax])),
    x2="x1:Q",
    y=alt.value(0),
    y2=alt.value(120),   
    color=alt.Color("band:N", legend=alt.Legend(title="Risk band"),
                    scale=alt.Scale(range=bands["color"].tolist()))
).properties(height=160)

zero_rule = alt.Chart(pd.DataFrame({"x":[0]})).mark_rule(color="#888", strokeDash=[4,4]).encode(x="x:Q")

# green if >0, red if <0, gray if 0
dot_color = "#10b981" if pct > 0 else "#ef4444" if pct < 0 else "#6b7280"
point_df = pd.DataFrame({"x":[pct], "label":[f"{pct:+.1f}%"]})
point = alt.Chart(point_df).mark_point(size=250, color=dot_color).encode(
    x="x:Q", y=alt.value(60)
)
label = alt.Chart(point_df).mark_text(dy=-22, fontWeight="bold").encode(
    x="x:Q", y=alt.value(60), text="label:N"
)

st.caption("Change vs latest (%). Bands show rough risk zones; your dot shows the scenario change.")
st.altair_chart(bands_chart + zero_rule + point + label, use_container_width=True)
