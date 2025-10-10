# Screening Simulation

An interactive web simulator for multi-stage candidate screening processes.  
Explore how **strictness**, **capacity limits**, and **accuracy trade-offs** shape your final hires, error rates, and mistake breakdowns.

🔗 Live demo: [https://screening-simulation.streamlit.app/](https://screening-simulation.streamlit.app/)

---

## 🎯 Purpose

This tool helps hiring teams and data analysts understand:

- How changing **screening strictness** shifts false positives (FP) vs false negatives (FN)  
- How bottlenecks and capacity constraints degrade screening accuracy  
- Where mistakes predominantly occur (CV review, technical interview, hiring manager step)  
- The trade-offs between processing more candidates vs maintaining quality

---

## 🧩 How It Works

The simulation models three consecutive stages:

1. **CV Screening** (low barrier, large capacity)  
2. **Technical Interview**  
3. **Hiring Manager Interview** (final decision)

Each stage has:

- **TPR** (True Positive Rate): probability a qualified candidate passes  
- **TNR** (True Negative Rate): probability an unqualified candidate is correctly rejected  
- **Capacity**: max number of candidates the stage can evaluate  
- **Optional “pressure” effect**: if the stage is overloaded, TPR and TNR degrade

You pick a **strictness parameter**, which linearly interpolates between lenient vs strict endpoint accuracies for each stage.

---

## ⚙️ Controls & Inputs

Use the left sidebar to tweak:

| Parameter | Meaning |
|-----------|---------|
| **p_good** | Proportion of qualified candidates in the initial pool |
| **N** | Total number of candidates sent through the pipeline |
| **Capacities** | Max throughput at CV, Tech, HM stages |
| **Strictness Sweep** | Define the strictness range and step for the simulation |
| **TPR/TNR endpoints** | Accuracy settings at lenient and strict extremes for each stage |
| **Capacity pressure toggle** | Enable/disable performance degradation under overload |
| **u_thr / u_max** | Utilization thresholds for pressure effect (see below) |
| **α_TPR / α_TNR** | How much accuracy (TPR, TNR) can drop under max pressure |

---

### 🔍 Strictness Sweep

Instead of simulating just one strictness value, the app sweeps across a *range* and plots the results:

- **Range (min, max)** → defines the span of strictness values.  
  - Lowering the **minimum** → includes more *lenient* scenarios (closer to “everyone passes”).  
  - Raising the **maximum** → includes more *strict* scenarios (closer to “almost no one passes”).  

- **Step** → defines how finely the range is divided.  
  - Smaller step = smoother curves (more data points, slightly slower).  
  - Larger step = coarser curves (fewer data points, faster).  

👉 Example:  
- Sweep from **0.0 → 1.0** with step **0.05** = 21 simulations, full range with good resolution.  
- Sweep from **0.3 → 0.8** with step **0.1** = narrower range, 6 simulations only.

---

### ⚡ Capacity Pressure (`u_thr` and `u_max`)

The pressure effect models what happens when a stage is overloaded:

- **Utilization (u)** = candidates arriving ÷ stage capacity.  
- **`u_thr` (threshold)** = the utilization level where pressure begins.  
  - If \( u \leq u_{thr} \), no penalty (TPR/TNR at full strength).  
  - Example: if `u_thr = 0.9` and CV capacity = 400, then as long as ≤360 candidates arrive, accuracy is unaffected.  

- **`u_max` (maximum)** = the point where pressure penalty reaches its worst case.  
  - Between `u_thr` and `u_max`, accuracy degrades linearly.  
  - At `u_max`, the full penalty (defined by `α_TPR` and `α_TNR`) is applied.  
  - If \( u > u_{max} \), penalty stays capped — accuracy cannot get worse beyond this.  

👉 Example:  
- `u_thr = 0.9`, `u_max = 1.5`, `α_TPR = 0.3`.  
- At \( u = 1.0 \) → slight penalty, TPR drops ~5%.  
- At \( u = 1.5 \) → full penalty, TPR drops 30%.  
- At \( u = 2.0 \) → penalty remains at 30% (capped).  

---

## 📊 Charts & Outputs

- **Summary KPIs**: Good hires, Bad hires, Precision, Final TPR  
- **Good & Bad Hires — Before vs After Pressure**: Compare results with and without overload effects  
-- **Per-Stage FN–FP Trade-Off (tabbed chart)**: Explore false negatives and false positives for each stage (CV, Tech, HM, Final) using interactive tabs  
- **Stage-level FN & FP breakdown**: Mistakes introduced in each stage, stacked for comparison  
- **Raw data table**: Full numerical results for each strictness value  

---

## 🚀 Try It Yourself

1. Adjust the sliders to explore different scenarios.  
2. Observe how hires and error rates shift.  
3. Toggle pressure on/off to see its effect.  
4. Experiment with extreme strictness or low capacity to test robustness.  

---

## 💡 Tips for Use

- **If capacity is far larger than demand**, the pressure effect won’t activate — the simulator behaves like the classic trade-off model.  
- **High α_TPR or α_TNR** = you lose accuracy quickly under overload (useful to model burnout or overwork).  
- Use the charts together: the FN vs FP curve shows trade-off, while stage-level stacks show *where* mistakes arise.

---

## 🛠️ Deployment & Code

This is a [Streamlit](https://streamlit.io/) app. The repo contains:

- `app.py` — main application code  
- `requirements.txt` — required Python packages  
- `README.md` — this file  

To run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
