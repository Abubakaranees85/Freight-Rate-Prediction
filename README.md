# Freight Rate ML Assessment

## Setup

```bash
pip install -r requirements.txt
pip install scikit-learn
```

## Run

Train the model and generate predictions:

```bash
python solution.py
```

This produces:
- `validation_predictions.csv` — 12,000 predicted rates for the validation set
- `december-chart-inputs.csv` — filled with December 2025 predictions

Then validate and generate the December chart:

```bash
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv --output-dir scorer_results
```

Chart is saved to `scorer_results/candidate_december.png`.

## Files

| File | Description |
|------|-------------|
| `train-test.csv` | Labeled training data (48,000 rows) |
| `validation.csv` | Unlabeled validation set (12,000 rows) |
| `december-chart-inputs.csv` | Fixed December lane inputs + predicted rates |
| `validation_predictions.csv` | Final predictions output |
| `solution.py` | Full pipeline: feature engineering, training, inference |
| `score.py` | Provided validation and chart generation script |
