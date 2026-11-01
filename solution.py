from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── load data ────────────────────────────────────────────────────────────────
train_df = pd.read_csv("train-test.csv")
val_df   = pd.read_csv("validation.csv")
dec_df   = pd.read_csv("december-chart-inputs.csv")


# ── feature engineering ──────────────────────────────────────────────────────
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["month"]      = df["date"].dt.month
    df["dayofweek"]  = df["date"].dt.dayofweek
    df["dayofyear"]  = df["date"].dt.dayofyear
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # haversine distance (sanity / extra signal)
    lat1 = np.radians(df["pickup_lat"])
    lat2 = np.radians(df["delivery_lat"])
    lon1 = np.radians(df["pickup_lon"])
    lon2 = np.radians(df["delivery_lon"])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    df["hav_dist"] = 3959 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    df["weight"] = df["weight"].fillna(df["weight"].median())
    df["market_index"] = df["market_index"].fillna(df["market_index"].median())

    df["equipment"] = df["equipment"].str.strip()
    return df


train_df = engineer(train_df)
val_df   = engineer(val_df)

# build december rows using train medians for weight/market_index (already filled)
dec_df["date"]         = pd.to_datetime(dec_df["date"])
dec_df["pickup_lat"]   = train_df.loc[train_df["pickup"] == "Lexington",  "pickup_lat"].iloc[0]
dec_df["pickup_lon"]   = train_df.loc[train_df["pickup"] == "Lexington",  "pickup_lon"].iloc[0]
dec_df["delivery_lat"] = train_df.loc[train_df["delivery"] == "Fort Wayne", "delivery_lat"].iloc[0]
dec_df["delivery_lon"] = train_df.loc[train_df["delivery"] == "Fort Wayne", "delivery_lon"].iloc[0]
dec_df["market_index"] = train_df["market_index"].median()
dec_df["quote_signal"] = train_df["quote_signal"].median()
dec_df = engineer(dec_df)

# ── encode categoricals ──────────────────────────────────────────────────────
from sklearn.preprocessing import LabelEncoder  # noqa: E402

for col in ["pickup", "delivery", "equipment"]:
    le = LabelEncoder()
    le.fit(pd.concat([train_df[col], val_df[col], dec_df[col]]))
    train_df[col] = le.transform(train_df[col])
    val_df[col]   = le.transform(val_df[col])
    dec_df[col]   = le.transform(dec_df[col])

FEATURES = [
    "pickup", "delivery", "pickup_lat", "pickup_lon",
    "delivery_lat", "delivery_lon", "distance", "hav_dist",
    "equipment", "weight", "market_index", "quote_signal",
    "month", "dayofweek", "dayofyear", "is_weekend",
]

X      = train_df[FEATURES]
y      = train_df["posted_rate"]
X_val  = val_df[FEATURES]
X_dec  = dec_df[FEATURES]

# ── train / validation split ──────────────────────────────────────────────────
from sklearn.model_selection import train_test_split  # noqa: E402

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)

# ── model ─────────────────────────────────────────────────────────────────────
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.metrics import mean_absolute_error, r2_score  # noqa: E402

model = GradientBoostingRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=10,
    subsample=0.8,
    random_state=SEED,
)
model.fit(X_tr, y_tr)

preds_te = model.predict(X_te)
mae = mean_absolute_error(y_te, preds_te)
r2  = r2_score(y_te, preds_te)
print(f"Holdout MAE: {mae:.2f}  |  R²: {r2:.4f}")

# ── validation predictions ────────────────────────────────────────────────────
val_preds = model.predict(X_val)
val_preds = np.maximum(val_preds, 1.0)   # ensure positive

out = pd.DataFrame({"load_id": val_df["load_id"], "predicted_rate": val_preds})
out.to_csv("validation_predictions.csv", index=False)
print(f"Saved validation_predictions.csv ({len(out)} rows)")

# ── december predictions ──────────────────────────────────────────────────────
dec_preds = model.predict(X_dec)
dec_preds = np.maximum(dec_preds, 1.0)

dec_out = pd.read_csv("december-chart-inputs.csv")
dec_out["predicted_rate"] = dec_preds
dec_out.to_csv("december-chart-inputs.csv", index=False)
print("Saved december-chart-inputs.csv with predicted_rate filled")
