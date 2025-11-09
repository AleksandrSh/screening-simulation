# app.py
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Screening Simulation", layout="wide")

# =========================================================
# Sidebar Controls
# =========================================================
st.title("Candidates Screening Simulation")
st.caption("Capacity-first pipeline with asymmetric strictness and capacity-pressure effects")
st.info("💡 **Terminology Guide**: Hover over controls and metrics for explanations. **TP** = True Positives (correctly accepted qualified candidates), **FP** = False Positives (incorrectly accepted unqualified candidates), **FN** = False Negatives (incorrectly rejected qualified candidates), **TN** = True Negatives (correctly rejected unqualified candidates)")

with st.sidebar:
    st.header("Inputs")

    # Base prevalence & pool size
    p_good = st.slider("Prevalence of qualified (p_good)", 0.01, 0.80, 0.10, 0.01, 
                       help="The proportion of candidates in the initial pool who are actually qualified/good. For example, 0.10 means 10% of all candidates are qualified.")
    N = st.slider("Total candidates (N)", 100, 10000, 1000, 50,
                  help="The total number of candidates entering the screening pipeline at the start.")

    # Stage capacity limits
    st.subheader("Stage capacity limits")
    c1, c2, c3 = st.columns(3)
    with c1:
        Cap1 = st.number_input("CV", 1, 100000, 400, 1,
                               help="Maximum number of candidates that can be processed at the CV screening stage.")
    with c2:
        Cap2 = st.number_input("Tech", 1, 100000, 150, 1,
                               help="Maximum number of candidates that can be processed at the Technical interview stage.")
    with c3:
        Cap3 = st.number_input("HM", 1, 100000, 25, 1,
                               help="Maximum number of candidates that can be processed at the Hiring Manager interview stage.")

    # Strictness sweep
    st.subheader("Strictness sweep")
    s_min, s_max = st.slider("Strictness range [lenient..strict]", 0.0, 1.0, (0.0, 1.0), 0.05,
                             help="Range of strictness values to simulate. 0.0 = most lenient (high TPR, low TNR), 1.0 = most strict (low TPR, high TNR). The simulation will test multiple strictness levels within this range.")
    s_step = st.number_input("Step", 0.01, 1.0, 0.05, 0.01,
                            help="Step size between strictness values. Smaller steps give more detailed results but take longer to compute.")
    s_values = np.round(np.arange(s_min, s_max + 1e-9, s_step), 4)

    # Capacity pressure
    st.subheader("Capacity pressure")
    use_pressure = st.toggle("Enable capacity-pressure effect", value=True)
    st.caption("ℹ️ When enabled, performance degrades as stage utilization increases above the threshold. This simulates real-world effects where overloaded interviewers make worse decisions.")
    u_thr = st.slider("u_thr (pressure starts)", 0.50, 1.00, 0.90, 0.01,
                     help="Utilization threshold where capacity pressure begins. Above this level, interviewer performance starts to degrade.")
    u_max = st.slider("u_max (pressure saturates)", 1.00, 2.00, 1.50, 0.01,
                     help="Utilization level where capacity pressure reaches maximum effect. Performance degradation plateaus at this point.")
    alpha_tpr = st.slider("α_TPR (max TPR drop)", 0.0, 0.8, 0.30, 0.01,
                         help="Maximum reduction in True Positive Rate (TPR) due to capacity pressure. TPR = correctly identifying qualified candidates.")
    alpha_tnr = st.slider("α_TNR (max TNR drop)", 0.0, 0.8, 0.30, 0.01,
                         help="Maximum reduction in True Negative Rate (TNR) due to capacity pressure. TNR = correctly rejecting unqualified candidates.")

    # Asymmetric strictness (FP-averse)
    st.subheader("Asymmetric strictness (FP-averse)")
    use_asym = st.toggle("Enable over-cautious strictness (TNR ↑; TPR capped/penalized)", value=True)
    st.caption("ℹ️ When enabled, higher strictness increases TNR (better at rejecting bad candidates) but caps and penalizes TPR (worse at accepting good candidates). This simulates risk-averse hiring behavior.")
    caution_k = st.slider("Over-cautiousness k (TPR penalty per unit TNR gain)", 0.0, 2.0, 2.0, 0.05,
                         help="Controls how much TPR is penalized as TNR increases. Higher values mean more severe penalties for being overly cautious.")

    # FN–FP chart options (for pipeline view)
    st.subheader("FN–FP chart options")
    overload_stage = st.selectbox(
        "Overload segmentation stage (affects only Final FN–FP chart)",
        ["Stage1 (CV)", "Stage2 (Tech)", "Stage3 (HM)"],
        index=0,
        help="Which stage's utilization to use for color-coding the final FN-FP trade-off chart. This helps visualize how capacity pressure at different stages affects overall outcomes."
    )

    # Stage endpoints (lenient → strict)
    st.subheader("TPR/TNR endpoints (lenient → strict)")
    st.caption("💡 **TPR** = True Positive Rate (sensitivity): % of qualified candidates correctly accepted. **TNR** = True Negative Rate (specificity): % of unqualified candidates correctly rejected.")
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

    show_baseline_markers = st.sidebar.checkbox(
        "Show faint baseline markers (u_base)", value=False
    )

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
        # stage totals (for capacity analysis)
        "tot1": tot1, "tot2": tot2,
    }

def sweep(with_pressure):
    return pd.DataFrame([simulate_once(float(s), with_pressure) for s in s_values])

# =========================================================
# Run simulations
# =========================================================
# Save current settings
current_use_asym = use_asym
current_use_pressure = use_pressure

# Baseline: run with no effects enabled (true baseline)
use_asym = False
df_no = sweep(False)  # baseline with no effects at all

# Restore settings and run with effects enabled
use_asym = current_use_asym
df_yes = sweep(current_use_pressure)  # outcomes with current settings (asymmetric + pressure if enabled)

# =========================================================
# KPIs
# =========================================================
st.caption("**Note:** The numbers below are for the strictest setting (top strictness value).")
k1,k2,k3,k4 = st.columns(4)
with k1: st.metric("Good hires (s end, with pressure)", f"{df_yes['good'].iloc[-1]:.1f}",
                  help="Number of qualified candidates successfully hired after passing through all screening stages.")
with k2: st.metric("Bad hires (s end, with pressure)",  f"{df_yes['bad'].iloc[-1]:.1f}",
                  help="Number of unqualified candidates incorrectly hired after passing through all screening stages (these are costly mistakes).")
with k3: st.metric("Precision (s end, with pressure)",  f"{df_yes['precision'].iloc[-1]:.3f}",
                  help="Precision = TP / (TP + FP). The proportion of hired candidates who are actually qualified. Higher is better (fewer bad hires).")
with k4: st.metric("TPR_final (1 - FN rate)",           f"{1-df_yes['fn_rate'].iloc[-1]:.3f}",
                  help="True Positive Rate (Sensitivity) = TP / (TP + FN). The proportion of qualified candidates who were successfully hired. Higher is better (fewer missed good candidates).")

st.divider()

# =========================================================
# Charts (two columns)
# =========================================================
# Determine chart title based on enabled effects
if current_use_pressure and current_use_asym:
    chart_title = "Good & Bad Hires — Before vs After Effects (Pressure + Asymmetric Strictness)"
elif current_use_pressure:
    chart_title = "Good & Bad Hires — Before vs After Capacity Pressure"
elif current_use_asym:
    chart_title = "Good & Bad Hires — Before vs After Asymmetric Strictness"
else:
    chart_title = "Good & Bad Hires"

st.subheader(chart_title)
fig, ax = plt.subplots(figsize=(10,5))

# Check if any effect is enabled
any_effect_enabled = current_use_pressure or current_use_asym

if any_effect_enabled:
    # Show both "Before" and "After" lines when any effect is enabled
    ax.plot(df_no["s"],  df_no["good"], label="Good (Before)", linestyle="--", color="#1f77b4")
    ax.plot(df_no["s"],  df_no["bad"],  label="Bad (Before)", linestyle="--", color="#d62728")
    ax.plot(df_yes["s"], df_yes["good"], label="Good (After)", linewidth=2, color="#1f77b4")
    ax.plot(df_yes["s"], df_yes["bad"],  label="Bad (After)",  linewidth=2, color="#d62728")
else:
    # Show only "Before" lines when no effects are enabled
    ax.plot(df_no["s"],  df_no["good"], label="Good", linestyle="--", color="#1f77b4")
    ax.plot(df_no["s"],  df_no["bad"],  label="Bad", linestyle="--", color="#d62728")

ax.set_xlabel("Strictness s")
ax.set_ylabel("Hires")
ax.grid(True, alpha=0.3)
ax.legend(loc="best")
st.pyplot(fig)

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

# def plot_tradeoff_three_zones(ax, X, Y, util_base, u_thr, label_prefix, xlab, ylab):
#     """
#     Segments a trade-off curve into 3 zones using *baseline* utilization:
#       Zone A: u <= u_thr                 (pre-pressure)
#       Zone B: u_thr < u <= 1             (soft overload: penalty only)
#       Zone C: u > 1                      (hard overload: capacity exceeded)

#     Draws: solid line for A, dashdot for B, dashed for C.
#     Adds vertical lines at the first x of u=u_thr and u=1 (if present),
#     and shades the soft-overload and hard-overload x-ranges.
#     """
#     u = np.asarray(util_base)
#     X = np.asarray(X); Y = np.asarray(Y)

#     A = u <= u_thr
#     B = (u > u_thr) & (u <= 1.0)
#     C = u > 1.0

#     # ---- plotting
#     if A.any():
#         ax.plot(X[A], Y[A], linewidth=2, label=f"{label_prefix} Pre-pressure (u ≤ u_thr)")
#         ax.scatter(X[A], Y[A], s=12)

#     if B.any():
#         ax.plot(X[B], Y[B], linewidth=2, linestyle="dashdot",
#                 label=f"{label_prefix} Soft overload (u_thr < u ≤ 1)")
#         ax.scatter(X[B], Y[B], s=12)
#         # shade soft overload span on X
#         xb_min = float(np.nanmin(X[B])); xb_max = float(np.nanmax(X[B]))
#         if np.isfinite(xb_min) and np.isfinite(xb_max) and xb_max > xb_min:
#             ax.axvspan(xb_min, xb_max, alpha=0.10, label=f"{label_prefix} Soft overload region")

#     if C.any():
#         ax.plot(X[C], Y[C], linewidth=2, linestyle="--",
#                 label=f"{label_prefix} Hard overload (u > 1)")
#         ax.scatter(X[C], Y[C], s=12)
#         # shade hard overload span on X
#         xc_min = float(np.nanmin(X[C])); xc_max = float(np.nanmax(X[C]))
#         if np.isfinite(xc_min) and np.isfinite(xc_max) and xc_max > xc_min:
#             ax.axvspan(xc_min, xc_max, alpha=0.12, label=f"{label_prefix} Hard overload region")

#     # Vertical guides at first occurrences (map threshold-in-s to X via the first point in each mask)
#     # u = u_thr
#     if B.any():
#         thr_idx = int(np.argmax(B))  # first True in B
#         ax.axvline(X[thr_idx], linestyle=":", linewidth=2, label=f"{label_prefix} Penalty starts (u = u_thr)")

#     # u = 1
#     if C.any():
#         cap_idx = int(np.argmax(C))
#         ax.axvline(X[cap_idx], linestyle=":", linewidth=2, label=f"{label_prefix} Capacity crossing (u = 1)")

#     ax.set_xlabel(xlab); ax.set_ylabel(ylab)
#     ax.grid(True, alpha=0.3); ax.legend(loc="best")

# def plot_tradeoff_hybrid(
#     ax, X, Y,
#     util_base,     # baseline u (from df_no) → zones (A/B/C) & shading
#     util_actual,   # with-pressure u (from df_yes) → where overload REALLY starts
#     u_thr,
#     label_prefix,
#     xlab, ylab
# ):

#     X = np.asarray(X); Y = np.asarray(Y)
#     u_base = np.asarray(util_base)
#     u_act  = np.asarray(util_actual)

#     # -------- 1) ZONES from BASELINE (definition view)
#     pre_mask   = (u_base <= u_thr)                # Zone A: pre-pressure
#     soft_mask  = (u_base > u_thr) & (u_base <= 1) # Zone B: soft overload
#     hard_mask  = (u_base > 1.0)                   # Zone C: hard overload

#     # Draw each zone with distinct style + shaded spans (definition)
#     if pre_mask.any():
#         ax.plot(X[pre_mask], Y[pre_mask], lw=2, label=f"{label_prefix} Pre-pressure (u ≤ u_thr)")
#         ax.scatter(X[pre_mask], Y[pre_mask], s=14)

#     if soft_mask.any():
#         ax.plot(X[soft_mask], Y[soft_mask], lw=2, ls="dashdot",
#                 label=f"{label_prefix} Soft overload (u_thr < u ≤ 1)")
#         ax.scatter(X[soft_mask], Y[soft_mask], s=14)
#         xs, xe = float(np.nanmin(X[soft_mask])), float(np.nanmax(X[soft_mask]))
#         if np.isfinite(xs) and np.isfinite(xe) and xe > xs:
#             ax.axvspan(xs, xe, alpha=0.10, label=f"{label_prefix} Soft overload region")

#     if hard_mask.any():
#         ax.plot(X[hard_mask], Y[hard_mask], lw=2, ls="--",
#                 label=f"{label_prefix} Hard overload (u > 1)")
#         ax.scatter(X[hard_mask], Y[hard_mask], s=14)
#         xs, xe = float(np.nanmin(X[hard_mask])), float(np.nanmax(X[hard_mask]))
#         if np.isfinite(xs) and np.isfinite(xe) and xe > xs:
#             ax.axvspan(xs, xe, alpha=0.12, label=f"{label_prefix} Hard overload region")

#     # -------- 2) BASELINE vertical guides (definition)
#     # First index where u_base > u_thr (penalty starts by definition)
#     if soft_mask.any():
#         i_thr_base = int(np.argmax(soft_mask))
#         ax.axvline(X[i_thr_base], ls=":", lw=2, color=None, label=f"{label_prefix} Penalty starts (baseline u = u_thr)")
#     # First index where u_base > 1 (capacity crossing by definition)
#     if hard_mask.any():
#         i_cap_base = int(np.argmax(hard_mask))
#         ax.axvline(X[i_cap_base], ls=":", lw=2, color=None, label=f"{label_prefix} Capacity crossing (baseline u = 1)")

#     # -------- 3) ACTUAL markers (reality)
#     # Where the with-pressure utilization actually exceeds u_thr and 1.0
#     soft_act_mask = (u_act > u_thr) & (u_act <= 1.0)
#     hard_act_mask = (u_act > 1.0)

#     # first actual onset (u_act > u_thr)
#     if soft_act_mask.any():
#         i_thr_act = int(np.argmax(soft_act_mask))
#         ax.scatter([X[i_thr_act]], [Y[i_thr_act]], s=80, marker="^",
#                    label=f"{label_prefix} Penalty onset (actual u = u_thr)")
#         ax.axvline(X[i_thr_act], ls="-.", lw=1.8, alpha=0.7)

#     # first actual capacity crossing (u_act > 1)
#     if hard_act_mask.any():
#         i_cap_act = int(np.argmax(hard_act_mask))
#         ax.scatter([X[i_cap_act]], [Y[i_cap_act]], s=80, marker="v",
#                    label=f"{label_prefix} Capacity crossed (actual u = 1)")
#         ax.axvline(X[i_cap_act], ls="-.", lw=1.8, alpha=0.7)

#     # -------- 4) Cosmetics
#     ax.set_xlabel(xlab); ax.set_ylabel(ylab)
#     ax.grid(True, alpha=0.3)
#     ax.legend(loc="best")

def plot_tradeoff_actual(
    ax, X, Y,
    util_act,            # with-pressure utilization for this stage (df_yes["util*"])
    u_thr,
    label_prefix,
    xlab, ylab,
    # optional faint markers for baseline reference
    util_base=None,      # df_no["util*"] for the same stage (optional)
    show_base=False,
):
    """
    Segments the curve using ACTUAL utilization (consistent with current settings):
      - Pre-pressure: u_act ≤ u_thr          (solid)
      - Soft overload: u_thr < u_act ≤ 1     (dash-dot)
      - Hard overload: u_act > 1             (dashed)
    Optionally draws faint baseline (no-pressure) vertical markers for reference.
    """

    X = np.asarray(X); Y = np.asarray(Y)
    u = np.asarray(util_act)

    pre_mask  = (u <= u_thr)
    soft_mask = (u > u_thr) & (u <= 1.0)
    hard_mask = (u > 1.0)

    # --- Draw by actual masks
    if pre_mask.any():
        ax.plot(X[pre_mask], Y[pre_mask], lw=2, label=f"{label_prefix} Pre-pressure")
        ax.scatter(X[pre_mask], Y[pre_mask], s=12)

    if soft_mask.any():
        ax.plot(X[soft_mask], Y[soft_mask], lw=2, ls="dashdot",
                label=f"{label_prefix} Soft overload")
        ax.scatter(X[soft_mask], Y[soft_mask], s=12)
        xs, xe = float(np.nanmin(X[soft_mask])), float(np.nanmax(X[soft_mask]))
        if np.isfinite(xs) and np.isfinite(xe) and xe > xs:
            ax.axvspan(xs, xe, alpha=0.10, color="#cccccc", label=f"{label_prefix} Soft overload")

    if hard_mask.any():
        ax.plot(X[hard_mask], Y[hard_mask], lw=2, ls="--",
                label=f"{label_prefix} Hard overload")
        ax.scatter(X[hard_mask], Y[hard_mask], s=12)
        xs, xe = float(np.nanmin(X[hard_mask])), float(np.nanmax(X[hard_mask]))
        if np.isfinite(xs) and np.isfinite(xe) and xe > xs:
            ax.axvspan(xs, xe, alpha=0.12, label=f"{label_prefix} Hard overload")

    # # --- Actual vertical guides (first occurrences)
    # if soft_mask.any():
    #     i_thr_act = int(np.argmax(soft_mask))         # first point with u_act > u_thr
    #     ax.axvline(X[i_thr_act], ls="-.", lw=1.8, alpha=0.7,
    #                label=f"{label_prefix} Penalty onset (actual u = u_thr)")
    # if hard_mask.any():
    #     i_cap_act = int(np.argmax(hard_mask))         # first point with u_act > 1
    #     ax.axvline(X[i_cap_act], ls="-.", lw=1.8, alpha=0.7,
    #                label=f"{label_prefix} Capacity crossed (actual u = 1)")

    # --- Optional: faint baseline markers for reference only
    if show_base and (util_base is not None):
        u0 = np.asarray(util_base)
        soft0 = (u0 > u_thr) & (u0 <= 1.0)
        hard0 = (u0 > 1.0)
        if soft0.any():
            i_thr_base = int(np.argmax(soft0))
            ax.axvline(X[i_thr_base], ls=":", lw=1.4, alpha=0.35,
                       label=f"{label_prefix} Baseline u = u_thr")
        if hard0.any():
            i_cap_base = int(np.argmax(hard0))
            ax.axvline(X[i_cap_base], ls=":", lw=1.4, alpha=0.35,
                       label=f"{label_prefix} Baseline u = 1")

    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3); ax.legend(loc="best")

st.subheader("Per-stage FN–FP Trade-offs")
st.caption("📊 **Stage Analysis**: Each tab shows the trade-off between False Negatives (missing good candidates) and False Positives (accepting bad candidates) at that specific screening stage. Red highlighting indicates when capacity constraints are causing candidate loss.")
rate_basis = st.radio("View", ["Stage rates (per seen good/bad)", "Counts"], horizontal=True,
                     help="**Rates**: Shows proportions relative to candidates seen at each stage. **Counts**: Shows absolute numbers of candidates.")

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
    # Add summary metrics above the chart
    st.caption("**Note:** The numbers below are for the strictest setting (top strictness value).")
    total_processed = df_yes["seen1"].iloc[-1]
    step_tp = df_yes["seen1_g"].iloc[-1] - df_yes["FN1"].iloc[-1]
    step_fp = df_yes["FP1"].iloc[-1]
    step_fn = df_yes["FN1"].iloc[-1]
    
    # Check if capacity is exceeded and calculate lost candidates
    capacity_exceeded = N > Cap1
    lost_total = max(0, N - total_processed)
    lost_good = lost_total * p_good if lost_total > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total processed (CV stage)", f"{total_processed:.0f}",
                 help="Number of candidates actually processed at the CV stage (limited by capacity).")
    with col2:
        st.metric("True Positives (CV)", f"{step_tp:.0f}",
                 help="Number of qualified candidates correctly accepted at the CV stage (good candidates passed through).")
    with col3:
        st.metric("False Positives (CV)", f"{step_fp:.0f}",
                 help="Number of unqualified candidates incorrectly accepted at the CV stage (bad candidates passed through).")
    with col4:
        if capacity_exceeded:
            st.markdown(
                f"""
                <div style="background-color: #ff4444; padding: 10px; border-radius: 5px; border: 2px solid #cc0000;">
                    <div style="color: white; font-weight: bold; font-size: 14px;">False Negatives (CV)</div>
                    <div style="color: white; font-size: 24px; font-weight: bold;">{step_fn:.0f}</div>
                    <div style="color: #ffcccc; font-size: 12px;">⚠️ {lost_good:.0f} potential TPs lost!</div>
                </div>
                """, 
                unsafe_allow_html=True,
                help=f"⚠️ Capacity exceeded! {lost_good:.0f} potential TP candidates lost due to capacity constraint ({lost_total:.0f} total candidates not processed)"
            )
        else:
            st.metric("False Negatives (CV)", f"{step_fn:.0f}",
                     help="Number of qualified candidates incorrectly rejected at the CV stage (good candidates filtered out).")

    fig_s1, ax_s1 = plt.subplots(figsize=(6,4))
    fp_rate_s1, fn_rate_s1, FP1, FN1 = stage_rates(df_yes, "seen1_g", "seen1_b", "FN1", "FP1")
    util_act_s1  = df_yes["util1"].to_numpy()   # ACTUAL segmentation
    util_base_s1 = df_no["util1"].to_numpy()    # optional faint markers

    if rate_basis.startswith("Stage"):
        X, Y = fp_rate_s1, fn_rate_s1
        xlab = "FP rate at CV (FP1 / seen1_b)"
        ylab = "FN rate at CV (FN1 / seen1_g)"
    else:
        X, Y = FP1, FN1
        xlab = "FP count at CV"
        ylab = "FN count at CV"

    plot_tradeoff_actual(ax_s1, X, Y, util_act_s1, u_thr, "CV", xlab, ylab,
                         util_base=util_base_s1, show_base=show_baseline_markers)

    # Add a second X axis at the top, showing per-point TP (rate or count) for each FP point
    ax_top = ax_s1.twiny()
    ax_top.set_xlim(ax_s1.get_xlim())
    TP_arr = df_yes["seen1_g"].to_numpy() - df_yes["FN1"].to_numpy()
    TP_rate_arr = np.where(df_yes["seen1_g"].to_numpy() > 0, TP_arr / df_yes["seen1_g"].to_numpy(), 0.0)
    fp_xticks = ax_s1.get_xticks()
    # For each FP tick, find the closest FP value in X and use its index for TP
    tick_indices = [int(np.argmin(np.abs(X - tick))) for tick in fp_xticks]
    if rate_basis.startswith("Stage"):
        top_labels = [f"{TP_rate_arr[i]:.2f}" for i in tick_indices]
        ax_top.set_xlabel("True Positive Rate (CV)", fontsize=10)
    else:
        top_labels = [f"{int(TP_arr[i])}" for i in tick_indices]
        ax_top.set_xlabel("True Positives (CV)", fontsize=10)
    ax_top.set_xticks(fp_xticks)
    ax_top.set_xticklabels(top_labels, rotation=0, fontsize=8)
    ax_top.xaxis.set_ticks_position('top')
    ax_top.xaxis.set_label_position('top')
    st.pyplot(fig_s1)

with tabs[1]:
    st.caption("**Note:** The numbers below are for the strictest setting (top strictness value).")
    total_processed2 = df_yes["seen2"].iloc[-1]
    step_tp2 = df_yes["seen2_g"].iloc[-1] - df_yes["FN2"].iloc[-1]
    step_fp2 = df_yes["FP2"].iloc[-1]
    step_fn2 = df_yes["FN2"].iloc[-1]
    
    # Check if capacity is exceeded and calculate lost candidates
    # For stage 2, input is tot1 (output from stage 1)
    stage1_total_output = df_yes["tot1"].iloc[-1]
    capacity_exceeded2 = stage1_total_output > Cap2
    lost_total2 = max(0, stage1_total_output - total_processed2)
    # Calculate good ratio from stage 1 output: (TP from stage 1) / (total output from stage 1)
    stage1_tp = df_yes["seen1_g"].iloc[-1] - df_yes["FN1"].iloc[-1]
    good_ratio_s1 = stage1_tp / stage1_total_output if stage1_total_output > 0 else 0
    lost_good2 = lost_total2 * good_ratio_s1 if lost_total2 > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total processed (Tech stage)", f"{total_processed2:.0f}",
                 help="Number of candidates actually processed at the Technical interview stage (limited by capacity).")
    with col2:
        st.metric("True Positives (Tech)", f"{step_tp2:.0f}",
                 help="Number of qualified candidates correctly accepted at the Technical interview stage.")
    with col3:
        st.metric("False Positives (Tech)", f"{step_fp2:.0f}",
                 help="Number of unqualified candidates incorrectly accepted at the Technical interview stage.")
    with col4:
        if capacity_exceeded2:
            st.markdown(
                f"""
                <div style="background-color: #ff4444; padding: 10px; border-radius: 5px; border: 2px solid #cc0000;">
                    <div style="color: white; font-weight: bold; font-size: 14px;">False Negatives (Tech)</div>
                    <div style="color: white; font-size: 24px; font-weight: bold;">{step_fn2:.0f}</div>
                    <div style="color: #ffcccc; font-size: 12px;">⚠️ {lost_good2:.0f} potential TPs lost!</div>
                </div>
                """, 
                unsafe_allow_html=True,
                help=f"⚠️ Capacity exceeded! {lost_good2:.0f} potential TP candidates lost due to capacity constraint ({lost_total2:.0f} total candidates not processed)"
            )
        else:
            st.metric("False Negatives (Tech)", f"{step_fn2:.0f}",
                     help="Number of qualified candidates incorrectly rejected at the Technical interview stage.")
    fig_s2, ax_s2 = plt.subplots(figsize=(6,4))
    fp_rate_s2, fn_rate_s2, FP2, FN2 = stage_rates(df_yes, "seen2_g", "seen2_b", "FN2", "FP2")
    util_act_s2  = df_yes["util2"].to_numpy()
    util_base_s2 = df_no["util2"].to_numpy()

    if rate_basis.startswith("Stage"):
        X, Y = fp_rate_s2, fn_rate_s2
        xlab = "FP rate at Tech (FP2 / seen2_b)"
        ylab = "FN rate at Tech (FN2 / seen2_g)"
    else:
        X, Y = FP2, FN2
        xlab = "FP count at Tech"
        ylab = "FN count at Tech"

    plot_tradeoff_actual(ax_s2, X, Y, util_act_s2, u_thr, "Tech", xlab, ylab,
                         util_base=util_base_s2, show_base=show_baseline_markers)
    # Add a second X axis at the top, showing per-point TP (rate or count) for each FP point
    ax_top2 = ax_s2.twiny()
    ax_top2.set_xlim(ax_s2.get_xlim())
    TP_arr2 = df_yes["seen2_g"].to_numpy() - df_yes["FN2"].to_numpy()
    TP_rate_arr2 = np.where(df_yes["seen2_g"].to_numpy() > 0, TP_arr2 / df_yes["seen2_g"].to_numpy(), 0.0)
    fp_xticks2 = ax_s2.get_xticks()
    tick_indices2 = [int(np.argmin(np.abs(X - tick))) for tick in fp_xticks2]
    if rate_basis.startswith("Stage"):
        top_labels2 = [f"{TP_rate_arr2[i]:.2f}" for i in tick_indices2]
        ax_top2.set_xlabel("True Positive Rate (Tech)", fontsize=10)
    else:
        top_labels2 = [f"{int(TP_arr2[i])}" for i in tick_indices2]
        ax_top2.set_xlabel("True Positives (Tech)", fontsize=10)
    ax_top2.set_xticks(fp_xticks2)
    ax_top2.set_xticklabels(top_labels2, rotation=0, fontsize=8)
    ax_top2.xaxis.set_ticks_position('top')
    ax_top2.xaxis.set_label_position('top')
    st.pyplot(fig_s2)

with tabs[2]:
    st.caption("**Note:** The numbers below are for the strictest setting (top strictness value).")
    total_processed3 = df_yes["seen3"].iloc[-1]
    step_tp3 = df_yes["seen3_g"].iloc[-1] - df_yes["FN3"].iloc[-1]
    step_fp3 = df_yes["FP3"].iloc[-1]
    step_fn3 = df_yes["FN3"].iloc[-1]
    
    # Check if capacity is exceeded and calculate lost candidates
    # For stage 3, input is tot2 (output from stage 2)
    stage2_total_output = df_yes["tot2"].iloc[-1]
    capacity_exceeded3 = stage2_total_output > Cap3
    lost_total3 = max(0, stage2_total_output - total_processed3)
    # Calculate good ratio from stage 2 output: (TP from stage 2) / (total output from stage 2)
    stage2_tp = df_yes["seen2_g"].iloc[-1] - df_yes["FN2"].iloc[-1]
    good_ratio_s2 = stage2_tp / stage2_total_output if stage2_total_output > 0 else 0
    lost_good3 = lost_total3 * good_ratio_s2 if lost_total3 > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total processed (HM stage)", f"{total_processed3:.0f}",
                 help="Number of candidates actually processed at the Hiring Manager interview stage (limited by capacity).")
    with col2:
        st.metric("True Positives (HM)", f"{step_tp3:.0f}",
                 help="Number of qualified candidates correctly accepted at the Hiring Manager interview stage.")
    with col3:
        st.metric("False Positives (HM)", f"{step_fp3:.0f}",
                 help="Number of unqualified candidates incorrectly accepted at the Hiring Manager interview stage.")
    with col4:
        if capacity_exceeded3:
            st.markdown(
                f"""
                <div style="background-color: #ff4444; padding: 10px; border-radius: 5px; border: 2px solid #cc0000;">
                    <div style="color: white; font-weight: bold; font-size: 14px;">False Negatives (HM)</div>
                    <div style="color: white; font-size: 24px; font-weight: bold;">{step_fn3:.0f}</div>
                    <div style="color: #ffcccc; font-size: 12px;">⚠️ {lost_good3:.0f} potential TPs lost!</div>
                </div>
                """, 
                unsafe_allow_html=True,
                help=f"⚠️ Capacity exceeded! {lost_good3:.0f} potential TP candidates lost due to capacity constraint ({lost_total3:.0f} total candidates not processed)"
            )
        else:
            st.metric("False Negatives (HM)", f"{step_fn3:.0f}",
                     help="Number of qualified candidates incorrectly rejected at the Hiring Manager interview stage.")
    fig_s3, ax_s3 = plt.subplots(figsize=(6,4))
    fp_rate_s3, fn_rate_s3, FP3, FN3 = stage_rates(df_yes, "seen3_g", "seen3_b", "FN3", "FP3")
    util_act_s3  = df_yes["util3"].to_numpy()
    util_base_s3 = df_no["util3"].to_numpy()

    if rate_basis.startswith("Stage"):
        X, Y = fp_rate_s3, fn_rate_s3
        xlab = "FP rate at HM (FP3 / seen3_b)"
        ylab = "FN rate at HM (FN3 / seen3_g)"
    else:
        X, Y = FP3, FN3
        xlab = "FP count at HM"
        ylab = "FN count at HM"

    plot_tradeoff_actual(ax_s3, X, Y, util_act_s3, u_thr, "HM", xlab, ylab,
                         util_base=util_base_s3, show_base=show_baseline_markers)
    # Add a second X axis at the top, showing per-point TP (rate or count) for each FP point
    ax_top3 = ax_s3.twiny()
    ax_top3.set_xlim(ax_s3.get_xlim())
    TP_arr3 = df_yes["seen3_g"].to_numpy() - df_yes["FN3"].to_numpy()
    TP_rate_arr3 = np.where(df_yes["seen3_g"].to_numpy() > 0, TP_arr3 / df_yes["seen3_g"].to_numpy(), 0.0)
    fp_xticks3 = ax_s3.get_xticks()
    tick_indices3 = [int(np.argmin(np.abs(X - tick))) for tick in fp_xticks3]
    if rate_basis.startswith("Stage"):
        top_labels3 = [f"{TP_rate_arr3[i]:.2f}" for i in tick_indices3]
        ax_top3.set_xlabel("True Positive Rate (HM)", fontsize=10)
    else:
        top_labels3 = [f"{int(TP_arr3[i])}" for i in tick_indices3]
        ax_top3.set_xlabel("True Positives (HM)", fontsize=10)
    ax_top3.set_xticks(fp_xticks3)
    ax_top3.set_xticklabels(top_labels3, rotation=0, fontsize=8)
    ax_top3.xaxis.set_ticks_position('top')
    ax_top3.xaxis.set_label_position('top')
    st.pyplot(fig_s3)

with tabs[3]:
    st.caption("**Note:** The numbers below are for the strictest setting (top strictness value).")
    total_intake = N
    true_positivesf = df_yes["good"].iloc[-1]
    false_positivesf = df_yes["bad"].iloc[-1]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Intake", f"{total_intake:.0f}",
                 help="Total number of candidates entering the entire screening pipeline.")
    with col2:
        st.metric("True Positives (final)", f"{true_positivesf:.0f}",
                 help="Number of qualified candidates who successfully passed through all stages and were hired.")
    with col3:
        st.metric("False Positives (final)", f"{false_positivesf:.0f}",
                 help="Number of unqualified candidates who incorrectly passed through all stages and were hired (bad hires).")
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

    if "Stage1" in overload_stage:
        util_act_final  = df_yes["util1"].to_numpy()
        util_base_final = df_no["util1"].to_numpy()
        prefix = "Final vs CV"
    elif "Stage2" in overload_stage:
        util_act_final  = df_yes["util2"].to_numpy()
        util_base_final = df_no["util2"].to_numpy()
        prefix = "Final vs Tech"
    else:
        util_act_final  = df_yes["util3"].to_numpy()
        util_base_final = df_no["util3"].to_numpy()
        prefix = "Final vs HM"

    plot_tradeoff_actual(ax_sf, X, Y, util_act_final, u_thr, prefix, xlab, ylab,
                         util_base=util_base_final, show_base=show_baseline_markers)
    # Add a second X axis at the top, showing per-point TP (rate or count) for each FP point
    ax_topf = ax_sf.twiny()
    ax_topf.set_xlim(ax_sf.get_xlim())
    TPf = df_yes["good"].to_numpy()
    TPf_rate = np.where(N * p_good > 0, TPf / (N * p_good), 0.0)
    fp_xticksf = ax_sf.get_xticks()
    tick_indicesf = [int(np.argmin(np.abs(X - tick))) for tick in fp_xticksf]
    if rate_basis.startswith("Stage"):
        top_labelsf = [f"{TPf_rate[i]:.2f}" for i in tick_indicesf]
        ax_topf.set_xlabel("True Positive Rate (Final)", fontsize=10)
    else:
        top_labelsf = [f"{int(TPf[i])}" for i in tick_indicesf]
        ax_topf.set_xlabel("True Positives (Final)", fontsize=10)
    ax_topf.set_xticks(fp_xticksf)
    ax_topf.set_xticklabels(top_labelsf, rotation=0, fontsize=8)
    ax_topf.xaxis.set_ticks_position('top')
    ax_topf.xaxis.set_label_position('top')
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