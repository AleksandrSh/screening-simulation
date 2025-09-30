
# Screening Simulation (Streamlit)

This app reproduces your Excel model with:
- Capacity-first pipeline (CV → Tech → HM)
- Strictness interpolation between lenient/strict TPR/TNR endpoints
- Optional capacity-pressure effect that degrades TPR/TNR when utilization exceeds thresholds
- Charts: before vs after pressure, FN–FP trade-off (with capacity marker), stage-level FN/FP stacks

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL (Streamlit will print it in the terminal).

## Deploy (free options)
- Streamlit Community Cloud (recommended): connect your repo and deploy.
- Hugging Face Spaces: add a Space with Streamlit SDK.
- Render / Fly.io: run `streamlit run app.py` on a free instance.

