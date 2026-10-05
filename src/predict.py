import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "best_model.joblib"
META = ROOT / "models" / "model_metadata.json"

def main(csv_path):
    if not MODEL.exists() or not META.exists():
        raise FileNotFoundError("Train Phase 1 first: python src/train_phase1.py")

    model = joblib.load(MODEL)
    with open(META, encoding="utf-8") as f:
        meta = json.load(f)

    df = pd.read_csv(csv_path)
    features = meta["feature_names"]
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    X = df[features].copy()
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]

    out = df[["URL"]].copy() if "URL" in df.columns else pd.DataFrame(index=df.index)
    out["prediction"] = ["phishing" if x == 1 else "legitimate" for x in pred]
    out["phishing_score"] = prob
    out.to_csv(ROOT / "reports" / "predictions.csv", index=False)
    print(out.head(20).to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()
    main(args.csv)
