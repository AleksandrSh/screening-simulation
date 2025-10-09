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
st.caption("Capacity-first pipeline with asymmetric strictness and capacity-pressure effects")

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

    # Asymmetric strictness (FP-averse)
    st.subheader("Asymmetric strictness (FP-averse)")
    use_asym = st.checkbox("Enable over-cautious strictness (TNR ↑; TPR capped/penalized)", value=True)
    caution_k = st.slider("Over-cautiousness k (TPR penalty per unit TNR gain)", 0.0, 2.0, 0.8, 0.05)

    # FN–FP chart options (for pipeline view)
    st.subheader("FN–FP chart options")
    overload_stage = st.selectbox(
        "Overload segmentation stage (for FN–FP chart)",
        ["Stage1 (CV)", "Stage2 (Tech)", "Stage3 (HM)"],
        index=1,
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
def lerp(a, b, s):
    return (1 - s) * a + s * b

def pressure(util, u_thr, u_max):
    return max(0.0, min(1.0, (util - u_thr) / max(1e-9, (u_max - u_thr))))

def asymmetric_rates(s, tpr_len, tpr_str, tnr_len, tnr_str, use_asym, k):
    """
    FP-averse strictness:
      - TNR rises with strictness (never below lenient).
      - TPR is capped at lenient and penalized as TNR rises.
    """
    tpr_lin = lerp(tpr_len, tpr_str, s)
    tnr_lin = lerp(tnr_len, tnr_str, s)

    if not use_asym:
        return tpr_lin, tnr_lin

    tnr_s = max(tnr_len, tnr_lin)
    gain = max(0.0, tnr_s - tnr_len)
    tpr_penalized = tpr_lin - k * gain
    tpr_s = min(tpr_len, tpr_penalized)
    tpr_s = max(0.0, min(1.0, tpr_s))
    return tpr_s, tnr_s

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
    return a_g, a_b, tot, fn, fp, util, seen, seen_g, seen_b

def next_stage(tot_prev, a_g_prev, a_b_prev, Cap, TPR, TNR, use_pressure, u_thr, u_max, a_tpr, a_tnr):
    seen = min(tot_prev, Cap)
    seen_g = seen * (a_g_prev / tot_prev) if tot_prev > 0 else 0.0
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
    return a_g, a_b, tot, fn, fp, util, seen, seen_g, seen_b

def simulate_once(s, with_pressure):
    # interpolate rates (asymmetric rule)
    TPR1, TNR1 = asymmetric_rates(s, TPR1_len, TPR1_str, TNR1_len, TNR1_str, use_asym, caution_k)
    TPR2, TNR2 = asymmetric_rates(s, TPR2_len, TPR2_str, TNR2_len, TNR2_str, use_asym, caution_k)
    TPR3, TNR3 = asymmetric_rates(s, TPR3_len, TPR3_str, TNR3_len, TNR3_str, use_asym, caution_k)

    # Stage 1
    a1_g, a1_b, tot1, fn1, fp1, util1, seen1, seen1_g, seen1_b = stage1(
        N, Cap1, p_good, TPR1, TNR1, with_pressure, u_thr, u_max, alpha_tpr, alpha_tnr
    )
    # Stage 2
    a2_g, a2_b, tot2, fn2, fp2, util2, seen2, seen2_g, seen2_b = next_stage(
        tot1, a1_g, a1_b, Cap2, TPR2, TNR2, with_pressure, u_thr, u_max, alpha_tpr, alpha_tnr
    )
    # Stage 3
    a3_g, a3_b, tot3, fn3, fp3, util3, seen3, seen3_g, seen3_b = next_stage(
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
        # stage mistakes
        "FN1": fn1, "FP1": fp1, "FN2": fn2, "FP2": fp2, "FN3": fn3, "FP3": fp3,
        # stage utilization
        "util1": util1, "util2": util2, "util3": util3,
        # stage seen bases
        "seen1": seen1, "seen1_g": seen1_g, "seen1_b": seen1_b,
        "seen2": seen2, "seen2_g": seen2_g, "seen2_b": seen2_b,
        "seen3": seen3, "seen3_g": seen3_g, "seen3_b": seen3_b,
    }

def sweep(with_pressure):
    return pd.DataFrame([simulate_once(float(s), with_pressure) for s in s_values])

# =========================================================
# Run simulations
# =========================================================
df_no  = sweep(False)               # baseline (no pressure) — used for segmentation
df_yes = sweep(True) if use_pressure else df_no.copy()  # outcomes shown

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
# Charts (two columns)
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

# --- Chart 2: FN–FP Trade-off (Pipeline) with baseline segmentation & rates/counts toggle
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

    # Counts (final outcomes)
    fp_final = df_yes["bad"].to_numpy()
    fn_final = (N * p_good - df_yes["good"]).to_numpy()

    # Counts (sum of stage mistakes)
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

    # Optional overlay: baseline (no-pressure) FN–FP curve (only for Rates)
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

# Keep the page aligned: put the explanation below the two-column block
with st.expander("ℹ️ How to read the FN–FP chart"):
    st.markdown(
        f"""
- **Segmentation** uses **baseline (no-pressure)** utilization of **{stage_label}** to decide normal (u ≤ 1) vs overload (u > 1).
- Curves/points show **with-pressure** outcomes for the selected view (rates or counts).
- The **vertical line** marks the first strictness where baseline utilization exceeds capacity (u = 1); the shaded area is the overload FP region.
"""
    )

# =========================================================
# NEW: Per-stage FN–FP Trade-offs (Rates or Counts)
# =========================================================
# def plot_segmented_tradeoff(ax, X, Y, util_base, label_prefix, xlab, ylab):
#     """
#     Draw a segmented FP–FN curve:
#       - util_base: baseline (no-pressure) utilization array for this stage (same length as X/Y)
#       - Normal: u ≤ 1 (solid)
#       - Overload: u > 1 (dashed)
#       - Capacity crossing vertical line + shaded overload band along X
#     """
#     normal_mask = util_base <= 1.0
#     over_mask   = util_base > 1.0

#     # Normal
#     if normal_mask.any():
#         ax.plot(X[normal_mask], Y[normal_mask], linewidth=2, label=f"{label_prefix} Normal (u ≤ 1)")
#         ax.scatter(X[normal_mask], Y[normal_mask], s=12)

#     # Overload
#     if over_mask.any():
#         ax.plot(X[over_mask], Y[over_mask], linewidth=2, linestyle="--", label=f"{label_prefix} Overload (u > 1)")
#         ax.scatter(X[over_mask], Y[over_mask], s=12)

#         # Capacity crossing index = first True in over_mask
#         cross_idx = int(np.argmax(over_mask))
#         ax.axvline(X[cross_idx], linestyle=":", linewidth=2, label=f"{label_prefix} Capacity crossing (u = 1)")

#         # Shade overload X range
#         x_min = float(np.nanmin(X[over_mask])); x_max = float(np.nanmax(X[over_mask]))
#         if np.isfinite(x_min) and np.isfinite(x_max) and x_max > x_min:
#             ax.axvspan(x_min, x_max, alpha=0.12, label=f"{label_prefix} Overload region")

#     ax.set_xlabel(xlab)
#     ax.set_ylabel(ylab)
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc="best")

def plot_tradeoff_three_zones(ax, X, Y, util_base, u_thr, label_prefix, xlab, ylab):
    """
    Segments a trade-off curve into 3 zones using *baseline* utilization:
      Zone A: u <= u_thr                 (pre-pressure)
      Zone B: u_thr < u <= 1             (soft overload: penalty only)
      Zone C: u > 1                      (hard overload: capacity exceeded)

    Draws: solid line for A, dashdot for B, dashed for C.
    Adds vertical lines at the first x of u=u_thr and u=1 (if present),
    and shades the soft-overload and hard-overload x-ranges.
    """
    u = np.asarray(util_base)
    X = np.asarray(X); Y = np.asarray(Y)

    A = u <= u_thr
    B = (u > u_thr) & (u <= 1.0)
    C = u > 1.0

    # ---- plotting
    if A.any():
        ax.plot(X[A], Y[A], linewidth=2, label=f"{label_prefix} Pre-pressure (u ≤ u_thr)")
        ax.scatter(X[A], Y[A], s=12)

    if B.any():
        ax.plot(X[B], Y[B], linewidth=2, linestyle="dashdot",
                label=f"{label_prefix} Soft overload (u_thr < u ≤ 1)")
        ax.scatter(X[B], Y[B], s=12)
        # shade soft overload span on X
        xb_min = float(np.nanmin(X[B])); xb_max = float(np.nanmax(X[B]))
        if np.isfinite(xb_min) and np.isfinite(xb_max) and xb_max > xb_min:
            ax.axvspan(xb_min, xb_max, alpha=0.10, label=f"{label_prefix} Soft overload region")

    if C.any():
        ax.plot(X[C], Y[C], linewidth=2, linestyle="--",
                label=f"{label_prefix} Hard overload (u > 1)")
        ax.scatter(X[C], Y[C], s=12)
        # shade hard overload span on X
        xc_min = float(np.nanmin(X[C])); xc_max = float(np.nanmax(X[C]))
        if np.isfinite(xc_min) and np.isfinite(xc_max) and xc_max > xc_min:
            ax.axvspan(xc_min, xc_max, alpha=0.12, label=f"{label_prefix} Hard overload region")

    # Vertical guides at first occurrences (map threshold-in-s to X via the first point in each mask)
    # u = u_thr
    if B.any():
        thr_idx = int(np.argmax(B))  # first True in B
        ax.axvline(X[thr_idx], linestyle=":", linewidth=2, label=f"{label_prefix} Penalty starts (u = u_thr)")

    # u = 1
    if C.any():
        cap_idx = int(np.argmax(C))
        ax.axvline(X[cap_idx], linestyle=":", linewidth=2, label=f"{label_prefix} Capacity crossing (u = 1)")

    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3); ax.legend(loc="best")

def plot_tradeoff_hybrid(
    ax, X, Y,
    util_base,     # baseline u (from df_no) → zones (A/B/C) & shading
    util_actual,   # with-pressure u (from df_yes) → where overload REALLY starts
    u_thr,
    label_prefix,
    xlab, ylab
):

    X = np.asarray(X); Y = np.asarray(Y)
    u_base = np.asarray(util_base)
    u_act  = np.asarray(util_actual)

    # -------- 1) ZONES from BASELINE (definition view)
    pre_mask   = (u_base <= u_thr)                # Zone A: pre-pressure
    soft_mask  = (u_base > u_thr) & (u_base <= 1) # Zone B: soft overload
    hard_mask  = (u_base > 1.0)                   # Zone C: hard overload

    # Draw each zone with distinct style + shaded spans (definition)
    if pre_mask.any():
        ax.plot(X[pre_mask], Y[pre_mask], lw=2, label=f"{label_prefix} Pre-pressure (u ≤ u_thr)")
        ax.scatter(X[pre_mask], Y[pre_mask], s=14)

    if soft_mask.any():
        ax.plot(X[soft_mask], Y[soft_mask], lw=2, ls="dashdot",
                label=f"{label_prefix} Soft overload (u_thr < u ≤ 1)")
        ax.scatter(X[soft_mask], Y[soft_mask], s=14)
        xs, xe = float(np.nanmin(X[soft_mask])), float(np.nanmax(X[soft_mask]))
        if np.isfinite(xs) and np.isfinite(xe) and xe > xs:
            ax.axvspan(xs, xe, alpha=0.10, label=f"{label_prefix} Soft overload region")

    if hard_mask.any():
        ax.plot(X[hard_mask], Y[hard_mask], lw=2, ls="--",
                label=f"{label_prefix} Hard overload (u > 1)")
        ax.scatter(X[hard_mask], Y[hard_mask], s=14)
        xs, xe = float(np.nanmin(X[hard_mask])), float(np.nanmax(X[hard_mask]))
        if np.isfinite(xs) and np.isfinite(xe) and xe > xs:
            ax.axvspan(xs, xe, alpha=0.12, label=f"{label_prefix} Hard overload region")

    # -------- 2) BASELINE vertical guides (definition)
    # First index where u_base > u_thr (penalty starts by definition)
    if soft_mask.any():
        i_thr_base = int(np.argmax(soft_mask))
        ax.axvline(X[i_thr_base], ls=":", lw=2, color=None, label=f"{label_prefix} Penalty starts (baseline u = u_thr)")
    # First index where u_base > 1 (capacity crossing by definition)
    if hard_mask.any():
        i_cap_base = int(np.argmax(hard_mask))
        ax.axvline(X[i_cap_base], ls=":", lw=2, color=None, label=f"{label_prefix} Capacity crossing (baseline u = 1)")

    # -------- 3) ACTUAL markers (reality)
    # Where the with-pressure utilization actually exceeds u_thr and 1.0
    soft_act_mask = (u_act > u_thr) & (u_act <= 1.0)
    hard_act_mask = (u_act > 1.0)

    # first actual onset (u_act > u_thr)
    if soft_act_mask.any():
        i_thr_act = int(np.argmax(soft_act_mask))
        ax.scatter([X[i_thr_act]], [Y[i_thr_act]], s=80, marker="^",
                   label=f"{label_prefix} Penalty onset (actual u = u_thr)")
        ax.axvline(X[i_thr_act], ls="-.", lw=1.8, alpha=0.7)

    # first actual capacity crossing (u_act > 1)
    if hard_act_mask.any():
        i_cap_act = int(np.argmax(hard_act_mask))
        ax.scatter([X[i_cap_act]], [Y[i_cap_act]], s=80, marker="v",
                   label=f"{label_prefix} Capacity crossed (actual u = 1)")
        ax.axvline(X[i_cap_act], ls="-.", lw=1.8, alpha=0.7)

    # -------- 4) Cosmetics
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

st.subheader("Per-stage FN–FP Trade-offs")
rate_basis = st.radio("View", ["Stage rates (per seen good/bad)", "Counts"], horizontal=True)

tabs = st.tabs(["CV (Stage1)", "Tech (Stage2)", "HM (Stage3)", "Final"])

def stage_rates(df, seen_g_col, seen_b_col, FN_col, FP_col):
    seen_g = df[seen_g_col].to_numpy()
    seen_b = df[seen_b_col].to_numpy()
    FN = df[FN_col].to_numpy()
    FP = df[FP_col].to_numpy()
    # Avoid divide-by-zero
    fn_rate = np.where(seen_g > 0, FN / seen_g, 0.0)
    fp_rate = np.where(seen_b > 0, FP / seen_b, 0.0)
    return fp_rate, fn_rate, FP, FN

with tabs[0]:
    fig_s1, ax_s1 = plt.subplots(figsize=(6,4))
    fp_rate_s1, fn_rate_s1, FP1, FN1 = stage_rates(df_yes, "seen1_g", "seen1_b", "FN1", "FP1")
    util_base_s1 = df_no["util1"].to_numpy()   # baseline (definition)
    util_act_s1  = df_yes["util1"].to_numpy()  # actual (reality)

    if rate_basis.startswith("Stage"):
        X, Y = fp_rate_s1, fn_rate_s1
        xlab = "FP rate at CV (FP1 / seen1_b)"
        ylab = "FN rate at CV (FN1 / seen1_g)"
    else:
        X, Y = FP1, FN1
        xlab = "FP count at CV"
        ylab = "FN count at CV"

    plot_tradeoff_hybrid(ax_s1, X, Y, util_base_s1, util_act_s1, u_thr, "CV", xlab, ylab)
    st.pyplot(fig_s1)

with tabs[1]:
    fig_s2, ax_s2 = plt.subplots(figsize=(6,4))
    fp_rate_s2, fn_rate_s2, FP2, FN2 = stage_rates(df_yes, "seen2_g", "seen2_b", "FN2", "FP2")
    util_base_s2 = df_no["util2"].to_numpy()
    util_act_s2  = df_yes["util2"].to_numpy()

    if rate_basis.startswith("Stage"):
        X, Y = fp_rate_s2, fn_rate_s2
        xlab = "FP rate at Tech (FP2 / seen2_b)"
        ylab = "FN rate at Tech (FN2 / seen2_g)"
    else:
        X, Y = FP2, FN2
        xlab = "FP count at Tech"
        ylab = "FN count at Tech"

    plot_tradeoff_hybrid(ax_s2, X, Y, util_base_s2, util_act_s2, u_thr, "Tech", xlab, ylab)
    st.pyplot(fig_s2)

with tabs[2]:
    fig_s3, ax_s3 = plt.subplots(figsize=(6,4))
    fp_rate_s3, fn_rate_s3, FP3, FN3 = stage_rates(df_yes, "seen3_g", "seen3_b", "FN3", "FP3")
    util_base_s3 = df_no["util3"].to_numpy()
    util_act_s3  = df_yes["util3"].to_numpy()

    if rate_basis.startswith("Stage"):
        X, Y = fp_rate_s3, fn_rate_s3
        xlab = "FP rate at HM (FP3 / seen3_b)"
        ylab = "FN rate at HM (FN3 / seen3_g)"
    else:
        X, Y = FP3, FN3
        xlab = "FP count at HM"
        ylab = "FN count at HM"

    plot_tradeoff_hybrid(ax_s3, X, Y, util_base_s3, util_act_s3, u_thr, "HM", xlab, ylab)
    st.pyplot(fig_s3)

with tabs[3]:
    fig_sf, ax_sf = plt.subplots(figsize=(6,4))
    FPf = df_yes["bad"].to_numpy()
    FNf = (N * p_good - df_yes["good"]).to_numpy()

    if rate_basis.startswith("Stage"):
        denom_bad  = N * (1 - p_good)
        denom_good = N * p_good
        X = np.where(denom_bad  > 0, FPf / denom_bad,  0.0)
        Y = np.where(denom_good > 0, FNf / denom_good, 0.0)
        xlab = "Final FP rate (bad / total bad)"
        ylab = "Final FN rate (missed / total good)"
    else:
        X, Y = FPf, FNf
        xlab = "Final FP count (bad hires)"
        ylab = "Final FN count (missed qualified)"

    # segmentation stage choice as in your sidebar
    if "Stage1" in overload_stage:
        util_base_final = df_no["util1"].to_numpy()
        util_act_final  = df_yes["util1"].to_numpy()
        prefix = "Final vs CV"
    elif "Stage2" in overload_stage:
        util_base_final = df_no["util2"].to_numpy()
        util_act_final  = df_yes["util2"].to_numpy()
        prefix = "Final vs Tech"
    else:
        util_base_final = df_no["util3"].to_numpy()
        util_act_final  = df_yes["util3"].to_numpy()
        prefix = "Final vs HM"

    plot_tradeoff_hybrid(ax_sf, X, Y, util_base_final, util_act_final, u_thr, prefix, xlab, ylab)
    st.pyplot(fig_sf)

# =========================================================
# Stage-level FN and FP (After Pressure)
# =========================================================
st.subheader("Stage-level FN and FP (After Pressure)")
fig3, (ax3, ax4) = plt.subplots(2, 1, figsize=(7,6), sharex=True)
ax3.stackplot(df_yes["s"], df_yes["FN1"], df_yes["FN2"], df_yes["FN3"],
              labels=["FN1 (CV)", "FN2 (Tech)", "FN3 (HM)"], alpha=0.8)
ax3.set_ylabel("FN count")
ax3.legend(loc="upper left")
ax3.grid(True, alpha=0.3)
ax4.stackplot(df_yes["s"], df_yes["FP1"], df_yes["FP2"], df_yes["FP3"],
              labels=["FP1 (CV)", "FP2 (Tech)", "FP3 (HM)"], alpha=0.8)
ax4.set_xlabel("Strictness s")
ax4.set_ylabel("FP count")
ax4.legend(loc="upper left")
ax4.grid(True, alpha=0.3)
st.pyplot(fig3)

st.divider()

# =========================================================
# Developer Debug View (optional) — keep as-is but compatible with your signature
# =========================================================
with st.expander("🛠️ Developer Debug View: Effective TPR/TNR curves"):
    fig_dbg, ax_dbg = plt.subplots(figsize=(7,4))

    # Raw linear interpolations for Stage 1 endpoints
    tpr1_raw = [lerp(TPR1_len, TPR1_str, s) for s in s_values]
    tnr1_raw = [lerp(TNR1_len, TNR1_str, s) for s in s_values]

    # After asymmetric penalty
    tpr1_asym_list = []
    tnr1_asym_list = []
    for s in s_values:
        tpr_s, tnr_s = asymmetric_rates(
            s, TPR1_len, TPR1_str, TNR1_len, TNR1_str, use_asym, caution_k
        )
        tpr1_asym_list.append(tpr_s)
        tnr1_asym_list.append(tnr_s)

    # After capacity pressure (use Stage 1 utilization from the WITH-pressure run)
    util_stage1 = df_yes["util1"].to_numpy()
    pres_vals = [pressure(u, u_thr, u_max) if use_pressure else 0.0 for u in util_stage1]

    tpr1_final = [t * (1 - alpha_tpr * p) for t, p in zip(tpr1_asym_list, pres_vals)]
    tnr1_final = [t * (1 - alpha_tnr * p) for t, p in zip(tnr1_asym_list, pres_vals)]

    ax_dbg.plot(s_values, tpr1_raw,  label="TPR (raw)",  linestyle="--")
    ax_dbg.plot(s_values, tnr1_raw,  label="TNR (raw)",  linestyle="--")
    ax_dbg.plot(s_values, tpr1_asym_list, label="TPR (asym)")
    ax_dbg.plot(s_values, tnr1_asym_list, label="TNR (asym)")
    ax_dbg.plot(s_values, tpr1_final, label="TPR (final, w/ pressure)", linewidth=2)
    ax_dbg.plot(s_values, tnr1_final, label="TNR (final, w/ pressure)", linewidth=2)

    ax_dbg.set_xlabel("Strictness s")
    ax_dbg.set_ylabel("Rate")
    ax_dbg.set_ylim(0, 1)
    ax_dbg.grid(True, alpha=0.3)
    ax_dbg.legend(loc="best")
    st.pyplot(fig_dbg)

    st.markdown("""
**How to read this debug chart:**
- **Raw** = linear interpolation between lenient and strict endpoints.
- **Asym** = after applying FP-averse strictness (TPR penalized as TNR rises).
- **Final** = after also applying capacity-pressure erosion (depends on utilization).
""")

st.subheader("Raw data (After Pressure)")
st.dataframe(df_yes.round(4))
