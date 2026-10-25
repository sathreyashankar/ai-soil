#!/usr/bin/env python3
"""
predict.py - Classify SFI_Class with the model from train.py and write SHAP-ready outputs.

Usage
    python predict.py                              # every row of SFI_final_dataset.csv
    python predict.py --only-test                  # only the held-out test rows (unseen by the model)
    python predict.py --input new_samples.csv      # any CSV that has the model's feature columns

Outputs (in --out-dir, default ./outputs)
    predictions.csv   Row_ID, Year/Sample_ID/Place, the feature values, predicted class,
                      class probabilities, and - if the input has SFI_Class - the true class,
                      a correct flag and (for the training dataset) which split each row was in
    X_shap.csv        only the model's feature columns, in training order, row-aligned with
                      predictions.csv -> the matrix to pass to shap.TreeExplainer(model)

Load for SHAP
    model = joblib.load("artifacts/lgbm_sfi_model.joblib")
    meta  = json.load(open("artifacts/model_metadata.json"))
    X     = pd.read_csv("outputs/X_shap.csv", index_col="Row_ID")[meta["feature_names"]]
    # SHAP output has one slice per class, in the order meta["class_order"] = Low, Medium, High

Requires: lightgbm>=4.6, scikit-learn, pandas, numpy, joblib
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict SFI_Class with the trained LightGBM model.")
    p.add_argument("--input", default="SFI_final_dataset.csv", help="CSV to classify")
    p.add_argument("--artifacts", default="artifacts", help="folder written by train.py")
    p.add_argument("--out-dir", default="outputs", help="where to write predictions.csv and X_shap.csv")
    p.add_argument(
        "--only-test",
        action="store_true",
        help="keep only the held-out test rows (input must be the dataset train.py used)",
    )
    return p.parse_args()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def align_columns(df: pd.DataFrame, wanted: list) -> pd.DataFrame:
    """Match column names ignoring surrounding spaces and case (e.g. 'PH' -> 'pH')."""
    df.columns = [str(c).strip() for c in df.columns]
    lookup = {c.lower(): c for c in df.columns}
    rename = {lookup[w.lower()]: w for w in wanted if w not in df.columns and w.lower() in lookup}
    return df.rename(columns=rename)


def split_labels(df: pd.DataFrame, art: Path) -> pd.Series:
    """train / test / excluded for each row of the dataset train.py used."""
    train_ids = pd.read_csv(art / "X_train.csv", usecols=["Row_ID"])["Row_ID"]
    test_ids = pd.read_csv(art / "X_test.csv", usecols=["Row_ID"])["Row_ID"]
    split = pd.Series("excluded", index=df.index, name="split")  # duplicates / invalid rows
    split[df.index.isin(train_ids)] = "train"
    split[df.index.isin(test_ids)] = "test"
    return split


def main() -> None:
    args = parse_args()
    art = Path(args.artifacts)
    in_path = Path(args.input)
    out = Path(args.out_dir)

    # ---- 1. Load model and metadata --------------------------------------------------
    meta_path = art / "model_metadata.json"
    if not meta_path.exists():
        raise SystemExit(f"{meta_path} not found - run train.py first (or pass --artifacts).")
    meta = json.loads(meta_path.read_text())
    model = joblib.load(art / meta["model_file"])
    features = meta["feature_names"]
    target = meta["target"]
    id_columns = meta["id_columns"]
    code_to_label = {int(k): v for k, v in meta["code_to_label"].items()}

    # ---- 2. Load input ---------------------------------------------------------------
    df = pd.read_csv(in_path, encoding="utf-8-sig")
    df = align_columns(df, features + [target] + id_columns)
    df.index.name = "Row_ID"  # 0-based row position in the input file
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise SystemExit(f"{in_path} is missing model feature columns: {missing}")

    X = df[features].apply(pd.to_numeric, errors="coerce").astype("float64")
    n_nan_rows = int(X.isna().any(axis=1).sum())
    if n_nan_rows:
        print(f"Warning: {n_nan_rows} rows have missing/non-numeric feature values "
              "(LightGBM still predicts them; check them before reading their SHAP values)")

    has_truth = target in df.columns
    same_dataset = file_sha256(in_path) == meta["data"]["sha256"]
    split = split_labels(df, art) if same_dataset else None

    if args.only_test:
        if split is None:
            raise SystemExit("--only-test needs the same dataset file train.py was trained on.")
        keep = split == "test"
        df, X, split = df[keep], X[keep], split[keep]

    # ---- 3. Predict ------------------------------------------------------------------
    proba = model.predict_proba(X)
    classes = [int(c) for c in model.classes_]
    pred_code = np.asarray(classes)[proba.argmax(axis=1)]
    pred_label = pd.Series(pred_code, index=X.index).map(code_to_label)

    parts = [df[[c for c in id_columns if c in df.columns]], X]
    if split is not None:
        parts.append(split)
    if has_truth:
        parts.append(df[target].astype(str).str.strip().rename(f"{target}_true"))
    parts.append(pred_label.rename(f"{target}_pred"))
    parts.append(pd.Series(pred_code, index=X.index, name=f"{target}_pred_code"))
    parts.append(pd.DataFrame(proba, index=X.index, columns=[f"prob_{code_to_label[c]}" for c in classes]))
    parts.append(pd.Series(proba.max(axis=1), index=X.index, name="confidence"))
    result = pd.concat(parts, axis=1)
    if has_truth:
        result["correct"] = result[f"{target}_true"] == result[f"{target}_pred"]

    # ---- 4. Save ---------------------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "predictions.csv")
    X.to_csv(out / "X_shap.csv")  # same rows, same order, model's column order

    # ---- 5. Report -------------------------------------------------------------------
    print(f"Classified {len(result)} rows from {in_path}")
    print("Predicted class counts:", result[f"{target}_pred"].value_counts().reindex(meta["class_order"]).fillna(0).astype(int).to_dict())
    if has_truth:
        truth, pred = result[f"{target}_true"], result[f"{target}_pred"]
        if split is not None and not args.only_test:
            for name in ["test", "train", "excluded"]:
                m = result["split"] == name
                if m.any():
                    note = "  <- honest score (unseen rows)" if name == "test" else ""
                    print(f"  {name:<9} {int(m.sum()):>6} rows  accuracy {accuracy_score(truth[m], pred[m]):.4f}{note}")
            print("  (train/excluded rows were seen during training, so their accuracy is optimistic)")
        else:
            print(f"Accuracy {accuracy_score(truth, pred):.4f} | macro F1 "
                  f"{f1_score(truth, pred, average='macro', labels=meta['class_order']):.4f}")
    print(f"\nWrote {out / 'predictions.csv'} and {out / 'X_shap.csv'}")
    print(f"SHAP: features = {features}; classes = {meta['class_order']}")


if __name__ == "__main__":
    main()
