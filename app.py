
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Screening Simulation", layout="wide")

# -----------------------
# Controls
# -----------------------
st.title("Screening Simulation")
st.caption("Capacity-first pipeline with strictness, TPR/TNR endpoints, and capacity-pressure effects")

with st.sidebar:
    st.header("Inputs")
    p_good = st.slider("Prevalence of qualified (p_good)", 0.01, 0.80, 0.10, 0.01)
    N = st.slider("Total candidates (N)", 100, 10000, 1000, 50)

    c1, c2, c3 = st.columns(3)
    Cap1 = c1.number_input("CV capacity", 1, 100000, 400, 1)
    Cap2 = c2.number_input("Tech capacity", 1, 100000, 75, 1)
    Cap3 = c3.number_input("HM capacity", 1, 100000, 25, 1)

    st.subheader("Strictness sweep")
    s_min, s_max = st.slider("Strictness range [lenient..strict]", 0.0, 1.0, (0.0, 1.0), 0.05)
    s_step = st.number_input("Step", 0.01, 1.0, 0.05, 0.01)
    s_values = np.round(np.arange(s_min, s_max + 1e-9, s_step), 4)

    st.subheader("Capacity pressure")
    use_pressure = st.toggle("Enable capacity-pressure effect", value=True)
    u_thr = st.slider("u_thr (start pressure)", 0.50, 1.00, 0.90, 0.01)
    u_max = st.slider("u_max (max pressure)", 1.00, 2.00, 1.50, 0.01)
    alpha_tpr = st.slider("α_TPR (max TPR drop)", 0.0, 0.8, 0.30, 0.01)
    alpha_tnr = st.slider("α_TNR (max TNR drop)", 0.0, 0.8, 0.30, 0.01)

    overload_stage = st.sidebar.selectbox(
    "Overload segmentation stage (for FN–FP chart)",
    ["Stage1 (CV)", "Stage2 (Tech)", "Stage3 (HM)"],
    index=1,  # default to Stage2 so you usually see both regions
)

    st.subheader("TPR/TNR endpoints (lenient → strict)")
    def rate_pair(label, tpr_len, tnr_len, tpr_str, tnr_str):
        c1,c2,c3,c4 = st.columns(4)
        tpr_len = c1.number_input(f"{label} TPR_len", 0.0, 1.0, tpr_len, 0.01, key=f"{label}_tprl")
        tnr_len = c2.number_input(f"{label} TNR_len", 0.0, 1.0, tnr_len, 0.01, key=f"{label}_tnrl")
        tpr_str = c3.number_input(f"{label} TPR_str", 0.0, 1.0, tpr_str, 0.01, key=f"{label}_tprs")
        tnr_str = c4.number_input(f"{label} TNR_str", 0.0, 1.0, tnr_str, 0.01, key=f"{label}_tnrs")
        return (tpr_len, tnr_len, tpr_str, tnr_str)

    TPR1_len, TNR1_len, TPR1_str, TNR1_str = rate_pair("CV",   0.50, 0.70, 0.85, 0.95)
    TPR2_len, TNR2_len, TPR2_str, TNR2_str = rate_pair("Tech", 0.60, 0.75, 0.90, 0.97)
    TPR3_len, TNR3_len, TPR3_str, TNR3_str = rate_pair("HM",   0.70, 0.80, 0.95, 0.98)

# -----------------------
# Model functions
# -----------------------
def lerp(a, b, s):
    return (1 - s) * a + s * b

def pressure(util, u_thr, u_max):
    return max(0.0, min(1.0, (util - u_thr) / max(1e-9, (u_max - u_thr))))

def stage1(N, Cap1, p_good, TPR1, TNR1, use_pressure, u_thr, u_max, alpha_tpr, alpha_tnr):
    seen = min(N, Cap1)
    seen_g = seen * p_good
    seen_b = seen - seen_g
    util = N / Cap1 if Cap1 > 0 else 99
    pres = pressure(util, u_thr, u_max) if use_pressure else 0.0
    tpr_eff = TPR1 * (1 - alpha_tpr * pres)
    tnr_eff = TNR1 * (1 - alpha_tnr * pres)
    a_g = seen_g * tpr_eff
    a_b = seen_b * (1 - tnr_eff)
    tot = a_g + a_b
    fn = seen_g - a_g
    fp = a_b
    return a_g, a_b, tot, fn, fp

def next_stage(tot_prev, a_g_prev, a_b_prev, Cap, TPR, TNR, use_pressure, u_thr, u_max, alpha_tpr, alpha_tnr):
    seen = min(tot_prev, Cap)
    if tot_prev > 0:
        seen_g = seen * (a_g_prev / tot_prev)
    else:
        seen_g = 0.0
    seen_b = seen - seen_g
    util = tot_prev / Cap if Cap > 0 else 99
    pres = pressure(util, u_thr, u_max) if use_pressure else 0.0
    tpr_eff = TPR * (1 - alpha_tpr * pres)
    tnr_eff = TNR * (1 - alpha_tnr * pres)
    a_g = seen_g * tpr_eff
    a_b = seen_b * (1 - tnr_eff)
    tot = a_g + a_b
    fn = seen_g - a_g
    fp = a_b
    return a_g, a_b, tot, fn, fp

def simulate_once(s, use_pressure):
    # interpolate rates
    TPR1 = lerp(TPR1_len, TPR1_str, s); TNR1 = lerp(TNR1_len, TNR1_str, s)
    TPR2 = lerp(TPR2_len, TPR2_str, s); TNR2 = lerp(TNR2_len, TNR2_str, s)
    TPR3 = lerp(TPR3_len, TPR3_str, s); TNR3 = lerp(TNR3_len, TNR3_str, s)

    # stage 1
    a1_g, a1_b, tot1, fn1, fp1 = stage1(N, Cap1, p_good, TPR1, TNR1, use_pressure, u_thr, u_max, alpha_tpr, alpha_tnr)
    # stage 2
    a2_g, a2_b, tot2, fn2, fp2 = next_stage(tot1, a1_g, a1_b, Cap2, TPR2, TNR2, use_pressure, u_thr, u_max, alpha_tpr, alpha_tnr)
    # stage 3 (final)
    a3_g, a3_b, tot3, fn3, fp3 = next_stage(tot2, a2_g, a2_b, Cap3, TPR3, TNR3, use_pressure, u_thr, u_max, alpha_tpr, alpha_tnr)

    good = a3_g
    bad  = a3_b
    total = good + bad
    precision = (good / total) if total > 0 else 0.0
    fp_rate = bad / N
    fn_rate = ((N * p_good - good) / (N * p_good)) if N * p_good > 0 else 0.0

    return {
        "s": s, "good": good, "bad": bad, "total": total,
        "precision": precision, "fp_rate": fp_rate, "fn_rate": fn_rate,
        "FN1": fn1, "FP1": fp1, "FN2": fn2, "FP2": fp2, "FN3": fn3, "FP3": fp3,
        "util1": N/Cap1 if Cap1>0 else 99,
        "util2": (tot1/Cap2) if Cap2>0 else 99,
        "util3": (tot2/Cap3) if Cap3>0 else 99,
    }

def sweep(use_pressure):
    return pd.DataFrame([simulate_once(float(s), use_pressure) for s in s_values])

# -----------------------
# Run simulation
# -----------------------
df_no  = sweep(False)
df_yes = sweep(True) if use_pressure else df_no.copy()

# -----------------------
# KPIs
# -----------------------
k1,k2,k3,k4 = st.columns(4)
with k1: st.metric("Good hires (s end, with pressure)", f"{df_yes['good'].iloc[-1]:.1f}")
with k2: st.metric("Bad hires (s end, with pressure)",  f"{df_yes['bad'].iloc[-1]:.1f}")
with k3: st.metric("Precision (s end, with pressure)",  f"{df_yes['precision'].iloc[-1]:.3f}")
with k4: st.metric("TPR_final (1 - FN)",               f"{1-df_yes['fn_rate'].iloc[-1]:.3f}")

st.divider()

# -----------------------
# Charts
# -----------------------
c1, c2 = st.columns(2)

with c1:
    st.subheader("Good & Bad Hires — Before vs After Pressure")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(df_no["s"],  df_no["good"], label="Good (Before)", linestyle="--")
    ax.plot(df_no["s"],  df_no["bad"],  label="Bad (Before)", linestyle="--")
    ax.plot(df_yes["s"], df_yes["good"], label="Good (After)", linewidth=2)
    ax.plot(df_yes["s"], df_yes["bad"],  label="Bad (After)", linewidth=2)
    ax.set_xlabel("Strictness s")
    ax.set_ylabel("Hires")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

with c2:
    st.subheader("FN vs FP Trade-off (Normal vs Overload)")

    # Pick which utilization series to use based on the selector
    if "Stage1" in overload_stage:
        util = df_yes["util1"].to_numpy()
    elif "Stage2" in overload_stage:
        util = df_yes["util2"].to_numpy()
    else:
        util = df_yes["util3"].to_numpy()

    fp = df_yes["fp_rate"].to_numpy()
    fn = df_yes["fn_rate"].to_numpy()

    # Masks
    normal_mask = util <= 1.0
    over_mask   = util > 1.0

    fig2, ax2 = plt.subplots(figsize=(6,4))

    # Plot normal and overload points/lines separately
    if normal_mask.any():
        ax2.plot(fp[normal_mask], fn[normal_mask], label="Normal (u ≤ 1)", linewidth=2)
        ax2.scatter(fp[normal_mask], fn[normal_mask], s=12)
    if over_mask.any():
        ax2.plot(fp[over_mask], fn[over_mask], label="Overload (u > 1)", linewidth=2, linestyle="--")
        ax2.scatter(fp[over_mask], fn[over_mask], s=12)

    # Dynamic vertical line at first crossing (first index where util > 1)
    cross_idx = np.argmax(over_mask) if over_mask.any() else None
    if cross_idx is not None and over_mask.any():
        x_cross = fp[cross_idx]
        ax2.axvline(x_cross, linestyle=":", linewidth=2, label="Capacity crossing (u=1)")

        # Shade overload region on FP axis (from first overload FP to max overload FP)
        x_min = np.nanmin(fp[over_mask])
        x_max = np.nanmax(fp[over_mask])
        if np.isfinite(x_min) and np.isfinite(x_max):
            ax2.axvspan(x_min, x_max, alpha=0.12, label="Overload region")

    # Labels and cosmetics
    ax2.set_xlabel("FP rate")
    ax2.set_ylabel("FN rate")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    # Caption clarifying which stage defines overload
    st.caption(
        f"Overload defined by **{overload_stage.split(' ')[0]}** utilization. "
        "Blue = normal; dashed/orange = overload; dotted line = first capacity crossing."
    )
    st.pyplot(fig2)

st.subheader("Stage-level FN and FP (After Pressure)")
fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(7,6), sharex=True)
ax3.stackplot(df_yes["s"], df_yes["FN1"], df_yes["FN2"], df_yes["FN3"], labels=["FN1 (CV)", "FN2 (Tech)", "FN3 (HM)"], alpha=0.8)
ax3.set_ylabel("FN count")
ax3.legend(loc="upper left")
ax3.grid(True, alpha=0.3)
ax4.stackplot(df_yes["s"], df_yes["FP1"], df_yes["FP2"], df_yes["FP3"], labels=["FP1 (CV)", "FP2 (Tech)", "FP3 (HM)"], alpha=0.8)
ax4.set_xlabel("Strictness s")
ax4.set_ylabel("FP count")
ax4.legend(loc="upper left")
ax4.grid(True, alpha=0.3)
st.pyplot(fig3)

st.divider()
st.subheader("Raw data (After Pressure)")
st.dataframe(df_yes.round(4))
