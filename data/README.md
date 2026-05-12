# Data

Place the Criteo CTR Prediction Dataset file here.

Expected raw file name:

```text
data/raw/criteo_train.txt
```

The original Criteo file is tab-separated with no header:

- Column 1: `label`
- Columns 2-14: integer features `I1` through `I13`
- Columns 15-40: categorical features `C1` through `C26`

For quick local experiments, you can also place a smaller sample at:

```text
data/sample/criteo_sample.txt
```
