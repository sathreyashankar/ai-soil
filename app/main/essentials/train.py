#!/usr/bin/env python3
"""
train.py - LightGBM classifier for the Soil Fertility Index class (SFI_Class: Low / Medium / High)

Trains on the raw soil-test parameters in SFI_final_dataset.csv and saves everything that
predict.py and a SHAP analysis need.

Usage
    python train.py
    python train.py --data SFI_final_dataset.csv --out-dir artifacts
    python train.py --keep-redundant      # keep N even though it is an exact copy of OC

Outputs (in --out-dir, default ./artifacts)
    lgbm_sfi_model.joblib      fitted LGBMClassifier -> joblib.load(), then shap.TreeExplainer(model)
    lgbm_sfi_model.txt         the same model in LightGBM's native text format
    model_metadata.json        feature order, class mapping, params, metrics, dataset fingerprint
    X_train.csv, y_train.csv   rows the final model was trained on   (SHAP background data)
    X_test.csv,  y_test.csv    held-out rows, never seen in training (rows to explain with SHAP)
    classification_report.txt, confusion_matrix.csv, confusion_matrix.png, feature_importance.csv

Requires: lightgbm>=4.6, scikit-learn, pandas, numpy, joblib, matplotlib
"""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
TARGET = "SFI_Class"
CLASS_ORDER = ["Low", "Medium", "High"]  # ordinal order -> encoded as 0, 1, 2
LABEL_TO_CODE = {label: code for code, label in enumerate(CLASS_ORDER)}

# The 12 soil-test parameters the SFI is computed from (model inputs).
SOIL_FEATURES = ["pH", "EC", "OC", "N", "P", "K", "S", "Zn", "B", "Fe", "Mn", "Cu"]

# Identifier columns: carried through to outputs for grouping, never used as model inputs.
ID_COLUMNS = ["Year", "Sample_ID", "Place"]

# Columns present in the dataset that must NOT be used as inputs, and why.
NORMALISED_COLUMNS = [f"N_{f}" for f in SOIL_FEATURES]  # N_pH, N_EC, ..., N_Cu
EXCLUDED_COLUMNS = {
    "SFI": "the continuous index SFI_Class is binned from (direct target leakage)",
    **{c: "normalised score inside the SFI weighted sum (target leakage)" for c in NORMALISED_COLUMNS},
    "N_raw_reported": "original lab N reading; not an input of the SFI formula",
    "Year": "identifier, not a soil property",
    "Sample_ID": "identifier",
    "Place": "location identifier; SFI depends only on the soil parameters",
}

VALIDATION_FRACTION = 0.15  # slice of the training rows used only to pick the number of trees
EARLY_STOPPING_ROUNDS = 100


def lgbm_params(seed: int) -> dict:
    """LightGBM hyper-parameters. n_estimators is an upper bound; early stopping picks the real value."""
    return dict(
        objective="multiclass",
        learning_rate=0.05,
        n_estimators=3000,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=1.0,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a LightGBM classifier for SFI_Class.")
    p.add_argument("--data", default="SFI_final_dataset.csv", help="path to SFI_final_dataset.csv")
    p.add_argument("--out-dir", default="artifacts", help="where to save the model and splits")
    p.add_argument("--test-size", type=float, default=0.20, help="held-out test fraction (default 0.20)")
    p.add_argument("--cv-folds", type=int, default=5, help="CV folds on the training rows (0 or 1 = skip)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--keep-redundant",
        action="store_true",
        help="keep features that are exact linear copies of another feature (e.g. N = 74.1 x OC)",
    )
    return p.parse_args()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in SOIL_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing required columns: {missing}")
    df.index.name = "Row_ID"  # 0-based row position in the CSV; lets every output be traced back
    for c in SOIL_FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET] = df[TARGET].astype(str).str.strip()
    return df


def find_redundant(X: pd.DataFrame, threshold: float = 0.9999) -> dict:
    """{dropped_feature: kept_feature} for features that are perfect linear copies of an earlier one."""
    corr = X.corr().abs()
    redundant = {}
    cols = list(X.columns)
    for i, a in enumerate(cols):
        if a in redundant:
            continue
        for b in cols[i + 1:]:
            if b not in redundant and corr.loc[a, b] >= threshold:
                redundant[b] = a
    return redundant


def save_xy(X: pd.DataFrame, y: pd.Series, out: Path, name: str) -> None:
    X.to_csv(out / f"X_{name}.csv")
    pd.DataFrame(
        {TARGET: y.map(dict(enumerate(CLASS_ORDER))), f"{TARGET}_code": y}, index=y.index
    ).to_csv(out / f"y_{name}.csv")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load and clean -----------------------------------------------------------
    df = load_dataset(data_path)
    n_raw = len(df)
    print(f"Loaded {n_raw} rows from {data_path}")

    valid = df[TARGET].isin(CLASS_ORDER) & df[SOIL_FEATURES].notna().all(axis=1)
    n_invalid = int((~valid).sum())
    if n_invalid:
        print(f"Dropped {n_invalid} rows with a missing soil value or a {TARGET} outside {CLASS_ORDER}")
    df = df[valid]

    # Identical samples in both train and test would inflate the test score.
    dup = df.duplicated(subset=SOIL_FEATURES + [TARGET], keep="first")
    n_dup = int(dup.sum())
    df = df[~dup]
    print(f"Removed {n_dup} exact duplicate samples -> {len(df)} unique rows")

    # ---- 2. Features -----------------------------------------------------------------
    features = list(SOIL_FEATURES)
    redundant = find_redundant(df[features])
    dropped_redundant = {}
    for dropped, kept in redundant.items():
        if args.keep_redundant:
            print(f"Note: '{dropped}' is an exact linear copy of '{kept}' (kept because of --keep-redundant)")
        else:
            print(
                f"'{dropped}' is an exact linear copy of '{kept}' -> dropped, so SHAP does not split one "
                f"effect across two identical columns ('{kept}' now carries both). Use --keep-redundant to keep it."
            )
            dropped_redundant[dropped] = kept
    features = [f for f in features if f not in dropped_redundant]
    print(f"Model features ({len(features)}): {features}")

    X = df[features].astype("float64")
    y = df[TARGET].map(LABEL_TO_CODE).astype(int).rename(f"{TARGET}_code")
    print("Class balance:", {CLASS_ORDER[k]: int(v) for k, v in y.value_counts().sort_index().items()})

    # ---- 3. Split --------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=VALIDATION_FRACTION, stratify=y_train, random_state=args.seed
    )
    print(f"Train {len(X_train)} rows | test {len(X_test)} rows (held out until the final evaluation)")

    # ---- 4. Pick the number of trees with early stopping ----------------------------
    params = lgbm_params(args.seed)
    probe = lgb.LGBMClassifier(**params)
    probe.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    best_iter = int(probe.best_iteration_ or params["n_estimators"])
    params["n_estimators"] = best_iter
    print(f"Early stopping chose {best_iter} trees")

    # ---- 5. Cross-validation on the training rows ------------------------------------
    cv_scores = []
    if args.cv_folds and args.cv_folds > 1:
        skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
        for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
            m = lgb.LGBMClassifier(**params).fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            acc = accuracy_score(y_train.iloc[va_idx], m.predict(X_train.iloc[va_idx]))
            cv_scores.append(float(acc))
            print(f"  CV fold {fold}/{args.cv_folds}: accuracy {acc:.4f}")
        print(f"CV accuracy: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")

    # ---- 6. Final model on all training rows -----------------------------------------
    model = lgb.LGBMClassifier(**params).fit(X_train, y_train)

    # ---- 7. Evaluate on the held-out test rows ---------------------------------------
    proba = model.predict_proba(X_test)
    y_pred = np.asarray(model.classes_)[proba.argmax(axis=1)]
    labels = list(range(len(CLASS_ORDER)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    low, high = LABEL_TO_CODE["Low"], LABEL_TO_CODE["High"]
    metrics = {
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "test_macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "test_log_loss": float(log_loss(y_test, proba, labels=labels)),
        "test_low_high_swaps": int(cm[low, high] + cm[high, low]),  # errors that skip a whole class
        "cv_accuracy_mean": float(np.mean(cv_scores)) if cv_scores else None,
        "cv_accuracy_std": float(np.std(cv_scores)) if cv_scores else None,
        "cv_fold_accuracies": cv_scores,
    }
    report = classification_report(y_test, y_pred, labels=labels, target_names=CLASS_ORDER, digits=4)

    print("\n=== Held-out test set ===")
    print(f"Accuracy          {metrics['test_accuracy']:.4f}")
    print(f"Balanced accuracy {metrics['test_balanced_accuracy']:.4f}")
    print(f"Macro F1          {metrics['test_macro_f1']:.4f}")
    print(f"Log loss          {metrics['test_log_loss']:.4f}")
    print(f"Low<->High swaps  {metrics['test_low_high_swaps']}")
    print(report)
    print("Confusion matrix (rows = true, cols = predicted):")
    print(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER))

    # ---- 8. Save everything ----------------------------------------------------------
    joblib.dump(model, out / "lgbm_sfi_model.joblib")
    model.booster_.save_model(str(out / "lgbm_sfi_model.txt"))

    save_xy(X_train, y_train, out, "train")
    save_xy(X_test, y_test, out, "test")

    (out / "classification_report.txt").write_text(report)
    pd.DataFrame(cm, index=[f"true_{c}" for c in CLASS_ORDER], columns=[f"pred_{c}" for c in CLASS_ORDER]).to_csv(
        out / "confusion_matrix.csv"
    )
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_ORDER).plot(ax=ax, cmap="Greens", values_format="d", colorbar=False)
    ax.set_title(f"LightGBM - SFI_Class (test accuracy {metrics['test_accuracy']:.3f})")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    booster = model.booster_
    importance = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": booster.feature_importance(importance_type="gain"),
            "split": booster.feature_importance(importance_type="split"),
        }
    )
    importance["gain_pct"] = 100 * importance["gain"] / importance["gain"].sum()
    importance = importance.sort_values("gain", ascending=False)
    importance.to_csv(out / "feature_importance.csv", index=False)

    metadata = {
        "model_file": "lgbm_sfi_model.joblib",
        "native_model_file": "lgbm_sfi_model.txt",
        "target": TARGET,
        "class_order": CLASS_ORDER,
        "label_to_code": LABEL_TO_CODE,
        "code_to_label": {str(c): lab for lab, c in LABEL_TO_CODE.items()},
        "feature_names": features,  # exact column order the model expects
        "id_columns": ID_COLUMNS,
        "dropped_redundant_features": dropped_redundant,
        "excluded_columns": EXCLUDED_COLUMNS,
        "params": params,
        "best_iteration": best_iter,
        "metrics": metrics,
        "data": {
            "path": str(data_path),
            "sha256": file_sha256(data_path),
            "rows_in_file": n_raw,
            "rows_invalid_dropped": n_invalid,
            "rows_duplicate_dropped": n_dup,
            "rows_used": int(len(X)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "test_size": args.test_size,
            "seed": args.seed,
        },
        "versions": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out / "model_metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"\nSaved model, splits and reports to {out.resolve()}")
    print("Top features by gain:", ", ".join(f"{r.feature} ({r.gain_pct:.1f}%)" for r in importance.head(5).itertuples()))
    print("Next: python predict.py")


if __name__ == "__main__":
    main()
