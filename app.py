# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Screening Simulation", layout="wide")

# =========================================================
# Sidebar Controls
# =========================================================
st.title("Screening Simulation")
st.caption("Capacity-first pipeline with strictness endpoints and capacity-pressure effects")

with st.sidebar:
    st.header("Inputs")

    # Base prevalence & pool size
    p_good = st.slider("Prevalence of qualified (p_good)", 0.01, 0.80, 0.10, 0.01)
    N = st.slider("Total candidates (N)", 100, 10000, 1000, 50)

    # Stage capacities
    c1, c2, c3 = st.columns(3)
    Cap1 = c1.number_input("CV capacity", 1, 100000, 400, 1)
    Cap2 = c2.number_input("Tech capacity", 1, 100000, 75, 1)
    Cap3 = c3.number_input("HM capacity", 1, 100000, 25, 1)

    # Strictness sweep
    st.subheader("Strictness sweep")
    s_min, s_max = st.slider("Strictness range [lenient..strict]", 0.0, 1.0, (0.0, 1.0), 0.05)
    s_step = st.number_input("Step", 0.01, 1.0, 0.05, 0.01)
    s_values = np.round(np.arange(s_min, s_max + 1e-9, s_step), 4)

    # Capacity pressure
    st.subheader("Capacity pressure")
    use_pressure = st.toggle("Enable capacity-pressure effect", value=True)
    u_thr = st.slider("u_thr (pressure starts)", 0.50, 1.00, 0.90, 0.01)
    u_max = st.slider("u_max (pressure saturates)", 1.00, 2.00, 1.50, 0.01)
    alpha_tpr = st.slider("α_TPR (max TPR drop)", 0.0, 0.8, 0.30, 0.01)
    alpha_tnr = st.slider("α_TNR (max TNR drop)", 0.0, 0.8, 0.30, 0.01)

    # Stage endpoints (lenient → strict)
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

    # FN–FP chart options
    st.subheader("FN–FP chart options")
    overload_stage = st.selectbox(
        "Overload segmentation stage (for FN–FP chart)",
        ["Stage1 (CV)", "Stage2 (Tech)", "Stage3 (HM)"],
        index=1,  # default Stage2
    )
    mistake_view = st.radio(
        "Show FN–FP as",
        ["Rates", "Counts (final outcome)", "Counts (sum of stage mistakes)"],
        index=0
    )
    show_baseline_curve = st.checkbox("Also show FN–FP without pressure (overlay, rates only)", value=False)

# =========================================================
# Model Functions
# =========================================================
def lerp(a, b, s):  # linear interpolation
    return (1 - s) * a + s * b

def pressure(util, u_thr, u_max):
    # 0..1 penalty factor, linear between thresholds, capped
    return max(0.0, min(1.0, (util - u_thr) / max(1e-9, (u_max - u_thr))))

def stage1(N, Cap1, p_good, TPR1, TNR1, use_pressure, u_thr, u_max, a_tpr, a_tnr):
    seen = min(N, Cap1)
    seen_g = seen * p_good
    seen_b = seen - seen_g
    util = N / Cap1 if Cap1 > 0 else 99
    pres = pressure(util, u_thr, u_max) if use_pressure else 0.0
    tpr_eff = TPR1 * (1 - a_tpr * pres)
    tnr_eff = TNR1 * (1 - a_tnr * pres)
    a_g = seen_g * tpr_eff
    a_b = seen_b * (1 - tnr_eff)
    tot = a_g + a_b
    fn = seen_g - a_g
    fp = a_b
    return a_g, a_b, tot, fn, fp, util

def next_stage(tot_prev, a_g_prev, a_b_prev, Cap, TPR, TNR, use_pressure, u_thr, u_max, a_tpr, a_tnr):
    seen = min(tot_prev, Cap)
    if tot_prev > 0:
        seen_g = seen * (a_g_prev / tot_prev)
    else:
        seen_g = 0.0
    seen_b = seen - seen_g
    util = tot_prev / Cap if Cap > 0 else 99
    pres = pressure(util, u_thr, u_max) if use_pressure else 0.0
    tpr_eff = TPR * (1 - a_tpr * pres)
    tnr_eff = TNR * (1 - a_tnr * pres)
    a_g = seen_g * tpr_eff
    a_b = seen_b * (1 - tnr_eff)
    tot = a_g + a_b
    fn = seen_g - a_g
    fp = a_b
    return a_g, a_b, tot, fn, fp, util

def simulate_once(s, with_pressure):
    # interpolate rates
    TPR1 = lerp(TPR1_len, TPR1_str, s); TNR1 = lerp(TNR1_len, TNR1_str, s)
    TPR2 = lerp(TPR2_len, TPR2_str, s); TNR2 = lerp(TNR2_len, TNR2_str, s)
    TPR3 = lerp(TPR3_len, TPR3_str, s); TNR3 = lerp(TNR3_len, TNR3_str, s)

    # Stage 1
    a1_g, a1_b, tot1, fn1, fp1, util1 = stage1(
        N, Cap1, p_good, TPR1, TNR1, with_pressure, u_thr, u_max, alpha_tpr, alpha_tnr
    )
    # Stage 2
    a2_g, a2_b, tot2, fn2, fp2, util2 = next_stage(
        tot1, a1_g, a1_b, Cap2, TPR2, TNR2, with_pressure, u_thr, u_max, alpha_tpr, alpha_tnr
    )
    # Stage 3 (final)
    a3_g, a3_b, tot3, fn3, fp3, util3 = next_stage(
        tot2, a2_g, a2_b, Cap3, TPR3, TNR3, with_pressure, u_thr, u_max, alpha_tpr, alpha_tnr
    )

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
        "util1": util1, "util2": util2, "util3": util3
    }

def sweep(with_pressure):
    return pd.DataFrame([simulate_once(float(s), with_pressure) for s in s_values])

# =========================================================
# Run simulations
# =========================================================
df_no  = sweep(False)
df_yes = sweep(True) if use_pressure else df_no.copy()

# =========================================================
# KPIs
# =========================================================
k1,k2,k3,k4 = st.columns(4)
with k1: st.metric("Good hires (s end, with pressure)", f"{df_yes['good'].iloc[-1]:.1f}")
with k2: st.metric("Bad hires (s end, with pressure)",  f"{df_yes['bad'].iloc[-1]:.1f}")
with k3: st.metric("Precision (s end, with pressure)",  f"{df_yes['precision'].iloc[-1]:.3f}")
with k4: st.metric("TPR_final (1 - FN rate)",           f"{1-df_yes['fn_rate'].iloc[-1]:.3f}")

st.divider()

# =========================================================
# Charts
# =========================================================
c1, c2 = st.columns(2)

# --- Chart 1: Good & Bad Hires — Before vs After Pressure
with c1:
    st.subheader("Good & Bad Hires — Before vs After Pressure")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(df_no["s"],  df_no["good"], label="Good (Before)", linestyle="--")
    ax.plot(df_no["s"],  df_no["bad"],  label="Bad (Before)", linestyle="--")
    ax.plot(df_yes["s"], df_yes["good"], label="Good (After)", linewidth=2)
    ax.plot(df_yes["s"], df_yes["bad"],  label="Bad (After)",  linewidth=2)
    ax.set_xlabel("Strictness s")
    ax.set_ylabel("Hires")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    st.pyplot(fig)

# --- Chart 2: FN–FP Trade-off with baseline segmentation and counts/rates toggle
with c2:
    st.subheader("FN vs FP — Trade-off (Normal vs Overload)")

    # Baseline (no-pressure) utilization used only for segmentation
    if "Stage1" in overload_stage:
        util_base = df_no["util1"].to_numpy()
        stage_label = "Stage1 (CV)"
    elif "Stage2" in overload_stage:
        util_base = df_no["util2"].to_numpy()
        stage_label = "Stage2 (Tech)"
    else:
        util_base = df_no["util3"].to_numpy()
        stage_label = "Stage3 (HM)"

    # Values to plot (always with pressure)
    fp_rate = df_yes["fp_rate"].to_numpy()
    fn_rate = df_yes["fn_rate"].to_numpy()

    # Final-outcome counts
    fp_final = df_yes["bad"].to_numpy()
    fn_final = (N * p_good - df_yes["good"]).to_numpy()

    # Sum-of-stage mistakes
    fp_total_stages = (df_yes["FP1"] + df_yes["FP2"] + df_yes["FP3"]).to_numpy()
    fn_total_stages = (df_yes["FN1"] + df_yes["FN2"] + df_yes["FN3"]).to_numpy()

    if mistake_view == "Rates":
        X = fp_rate; Y = fn_rate; xlab = "FP rate"; ylab = "FN rate"
    elif mistake_view == "Counts (final outcome)":
        X = fp_final; Y = fn_final; xlab = "False Positives (bad hires)"; ylab = "False Negatives (missed qualified)"
    else:
        X = fp_total_stages; Y = fn_total_stages; xlab = "Total FP mistakes (all stages)"; ylab = "Total FN mistakes (all stages)"

    normal_mask = util_base <= 1.0
    over_mask   = util_base > 1.0

    fig2, ax2 = plt.subplots(figsize=(6,4))

    # Optional overlay: baseline (no-pressure) FN–FP curve (only meaningful for Rates)
    if mistake_view == "Rates" and show_baseline_curve:
        ax2.plot(
            df_no["fp_rate"], df_no["fn_rate"],
            alpha=0.5, linewidth=1.5, linestyle=":",
            label="FN–FP (no pressure)"
        )

    # Normal region (baseline u<=1)
    if normal_mask.any():
        ax2.plot(X[normal_mask], Y[normal_mask], label="Normal (u ≤ 1, baseline)", linewidth=2)
        ax2.scatter(X[normal_mask], Y[normal_mask], s=12)

    # Overload region (baseline u>1)
    if over_mask.any():
        ax2.plot(X[over_mask], Y[over_mask], label="Overload (u > 1, baseline)", linewidth=2, linestyle="--")
        ax2.scatter(X[over_mask], Y[over_mask], s=12)

        # Capacity crossing (first index where baseline util>1)
        cross_idx = int(np.argmax(over_mask))
        ax2.axvline(X[cross_idx], linestyle=":", linewidth=2, label="Capacity crossing (baseline u = 1)")

        # Shade overload range on X
        x_min = float(np.nanmin(X[over_mask])); x_max = float(np.nanmax(X[over_mask]))
        if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
            ax2.axvspan(x_min, x_max, alpha=0.12, label="Overload region (baseline)")

    ax2.set_xlabel(xlab)
    ax2.set_ylabel(ylab)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")
    st.pyplot(fig2)

# Put the explanatory note below both charts so columns align
st.caption(
    f"Segmentation uses **baseline (no-pressure)** utilization of {stage_label} to define normal vs overload. "
    "Curves/points show **with-pressure** outcomes. Use the selector to view **rates** or **counts**."
)

# --- Chart 3: Stage-level FN and FP (After Pressure)
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
