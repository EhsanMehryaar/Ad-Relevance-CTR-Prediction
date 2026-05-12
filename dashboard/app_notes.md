# Dashboard Notes

The root `app.py` is the Streamlit entrypoint. It reads model artifacts from `models/`:

- `model.joblib`
- `metrics.json`
- `feature_importance.csv`
- `calibration_curve.csv`
- `lift_chart.csv`

Run:

```bash
streamlit run app.py
```
