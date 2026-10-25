#!/usr/bin/env python3
"""
app.py - Streamlit dashboard for the SFI_Class LightGBM model (train.py / predict.py).

Run it from the folder that holds artifacts/ and SFI_final_dataset.csv:
    streamlit run app.py
    streamlit run app.py -- --artifacts artifacts --data SFI_final_dataset.csv

Tabs
    Overview           the dataset and how SFI_Class is defined                 (showcase)
    Model performance  how the model was trained, test metrics, confusion matrix (showcase)
    SHAP explanations  which soil parameters drive each class, and in which way  (showcase)
    Predict a sample   pick or type one sample -> class, probabilities and why   (testing)
    Batch prediction   upload a CSV -> predictions (+ SHAP values) to download   (testing)

SHAP values come from LightGBM's built-in TreeSHAP (model.predict(X, pred_contrib=True)),
the same exact Tree SHAP algorithm that shap.TreeExplainer uses for tree models.

Requires: streamlit>=1.40, lightgbm, scikit-learn, pandas, numpy, joblib (altair comes with streamlit)
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

alt.data_transformers.disable_max_rows()

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
SOIL_FEATURES = ["pH", "EC", "OC", "N", "P", "K", "S", "Zn", "B", "Fe", "Mn", "Cu"]
ID_COLUMNS = ["Year", "Sample_ID", "Place"]
MEDIAN_OPTION = "Typical sample (median of the training data)"
SHAP_SAMPLE_ROWS = 1000  # test rows used for the global SHAP charts (TreeSHAP is the slow part)

# Chart colours. Classes are ordered (Low < Medium < High), so they share one blue ramp
# that gets darker (light theme) / lighter (dark theme) with fertility. SHAP direction is
# a red/blue diverging pair. All sets were checked for colour-blind separation and contrast.
PALETTES = {
    "light": {
        "surface": "#ffffff",
        "classes": ["#86b6ef", "#2a78d6", "#104281"],
        "series": "#2a78d6",
        "toward": "#e34948",
        "away": "#2a78d6",
        "heat": ["#cde2fb", "#104281"],
        "value": ["#86b6ef", "#0d366b"],
        "text": "#52514e",
        "baseline": "#c3c2b7",
        "ink_on_dark": "#ffffff",
        "ink_on_light": "#0b0b0b",
    },
    "dark": {
        "surface": "#0e1117",
        "classes": ["#184f95", "#3987e5", "#9ec5f4"],
        "series": "#3987e5",
        "toward": "#e66767",
        "away": "#3987e5",
        "heat": ["#104281", "#cde2fb"],
        "value": ["#184f95", "#cde2fb"],
        "text": "#c3c2b7",
        "baseline": "#383835",
        "ink_on_dark": "#f0efec",
        "ink_on_light": "#0b0b0b",
    },
}


# --------------------------------------------------------------------------------------
# Paths and loading
# --------------------------------------------------------------------------------------
def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--data", default="SFI_final_dataset.csv")
    args, _ = p.parse_known_args(sys.argv[1:])
    return args


def clean_path(text: str) -> Path:
    """A path typed or pasted by the user (Windows 'Copy as path' adds quotes)."""
    return Path(text.strip().strip('"').strip("'")).expanduser()


def search_dirs(extra=()) -> list:
    """Where to look: the folder you started from, the folder app.py is in, and three parents of each."""
    out = []
    for base in [*extra, Path.cwd(), Path(__file__).resolve().parent]:
        base = Path(base).resolve()
        for d in [base, *list(base.parents)[:3]]:
            if d not in out:
                out.append(d)
    return out


def find_artifacts(name: str):
    """Return (artifacts folder or None, list of folders checked)."""
    p = Path(name).expanduser()
    if p.is_absolute():
        return (p if (p / "model_metadata.json").exists() else None), [p]
    checked = [d / p for d in search_dirs()]
    for c in checked:
        if (c / "model_metadata.json").exists():
            return c, checked
    # Not in the usual places: look one and two folders down; newest model wins.
    hits = set()
    for base in (Path.cwd().resolve(), Path(__file__).resolve().parent):
        for pattern in ("*/model_metadata.json", "*/*/model_metadata.json"):
            try:
                hits.update(base.glob(pattern))
            except OSError:
                pass
    if hits:
        return max(hits, key=lambda h: h.stat().st_mtime).parent, checked
    return None, checked


DATA_SUBFOLDERS = ("", "DataSets", "datasets", "Datasets", "dataset", "data", "Data")


def find_dataset(arg: str, default: str, meta: dict, art: Path):
    """Find the dataset, in this order:
    1. --data, if you passed it
    2. the file train.py was trained on (its path is stored in model_metadata.json, relative to
       the folder train.py was run from - normally the folder that holds artifacts/)
    3. a file with that name in the usual folders or a DataSets/data subfolder of them
    """
    tried = []
    if arg != default:
        p = Path(arg).expanduser()
        tried.append(p)
        if p.exists():
            return p, tried
    trained_on = Path(str(meta.get("data", {}).get("path", default))).expanduser()
    if trained_on.is_absolute():
        tried.append(trained_on)
        if trained_on.exists():
            return trained_on, tried
    else:
        for base in (art.parent, Path.cwd()):
            p = (base / trained_on).resolve()
            tried.append(p)
            if p.exists():
                return p, tried
    names = list(dict.fromkeys([Path(arg).name, trained_on.name]))
    for d in search_dirs(extra=(art.parent,)):
        for sub in DATA_SUBFOLDERS:
            for name in names:
                p = d / sub / name if sub else d / name
                if p.exists():
                    return p, tried
    tried += [d / name for d in search_dirs(extra=(art.parent,)) for name in names]
    return None, tried


def _remember_artifacts_path():
    st.session_state["artifacts_dir"] = st.session_state.get("artifacts_path_input", "")


def stamp(path: Path) -> str:
    """Changes whenever the file changes, so cached results refresh after retraining."""
    return f"{path.stat().st_mtime_ns}-{path.stat().st_size}" if path.exists() else "missing"


@st.cache_resource(show_spinner=False)
def load_model(artifacts: str, version: str):
    art = Path(artifacts)
    meta = json.loads((art / "model_metadata.json").read_text(encoding="utf-8"))
    model = joblib.load(art / meta.get("model_file", "lgbm_sfi_model.joblib"))
    return model, meta


@st.cache_data(show_spinner=False)
def load_split(artifacts: str, name: str, features: tuple, version: str):
    art = Path(artifacts)
    X = pd.read_csv(art / f"X_{name}.csv", index_col="Row_ID")[list(features)].astype("float64")
    y = pd.read_csv(art / f"y_{name}.csv", index_col="Row_ID")
    label_col = [c for c in y.columns if not c.endswith("_code")][0]
    return X, y[label_col].astype(str).reindex(X.index)


@st.cache_data(show_spinner=False)
def load_dataset(path: str, version: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    df.index.name = "Row_ID"
    return df


@st.cache_data(show_spinner=False)
def file_sha256(path: str, version: str) -> str:
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


# --------------------------------------------------------------------------------------
# Model helpers
# --------------------------------------------------------------------------------------
def class_labels(model, meta) -> list:
    """Class names in the order of the model's probability / SHAP columns."""
    code_to_label = {int(k): v for k, v in meta["code_to_label"].items()}
    return [code_to_label[int(c)] for c in model.classes_]


def predict(model, meta, X: pd.DataFrame):
    proba = model.predict_proba(X)
    labels = class_labels(model, meta)
    pred = np.asarray(labels)[proba.argmax(axis=1)]
    return proba, pred


def tree_shap(model, X: pd.DataFrame):
    """Exact Tree SHAP values.

    Returns values with shape (rows, features, classes) - the same layout shap.TreeExplainer
    gives for a multiclass model - and the base value of each class. Units are the model's
    raw score (log-odds); base + sum of a row's values = that row's raw score for the class.
    """
    n, f = X.shape
    k = len(model.classes_)
    contrib = np.asarray(model.predict(X, pred_contrib=True))
    if contrib.shape != (n, k * (f + 1)):
        raise ValueError(f"unexpected pred_contrib shape {contrib.shape}, expected {(n, k * (f + 1))}")
    arr = contrib.reshape(n, k, f + 1)
    return arr[:, :, :f].transpose(0, 2, 1), arr[0, :, f]


@st.cache_data(show_spinner="Scoring samples...")
def score(_model, _meta, version: str, X: pd.DataFrame):
    return predict(_model, _meta, X)


@st.cache_data(show_spinner="Computing SHAP values (only the first time)...", persist="disk")
def shap_for(_model, version: str, X: pd.DataFrame):
    return tree_shap(_model, X)


def ui_theme() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # st.context.theme needs Streamlit 1.46+
        return "light"


def pct(x) -> str:
    return "-" if x is None else f"{x:.1%}"


def when(iso: str) -> str:
    """'2026-10-17T17:55:25+00:00' -> '2026-10-17 17:55 UTC'"""
    return iso[:16].replace("T", " ") + (" UTC" if iso.endswith("+00:00") else "") if iso else "?"


# --------------------------------------------------------------------------------------
# Charts (Altair)
# --------------------------------------------------------------------------------------
def class_color(field: str, labels: list, pal: dict, legend=True):
    return alt.Color(
        f"{field}:N",
        scale=alt.Scale(domain=labels, range=pal["classes"]),
        sort=labels,
        legend=alt.Legend(title="SFI class", orient="top") if legend else None,
    )


def chart_samples_by_year(df: pd.DataFrame, target: str, labels: list, pal: dict):
    d = df.groupby(["Year", target]).size().rename("samples").reset_index()
    d["share"] = d["samples"] / d.groupby("Year")["samples"].transform("sum")
    d["order"] = d[target].map({c: i for i, c in enumerate(labels)})
    years = sorted(d["Year"].astype(str).unique())
    return (
        alt.Chart(d)
        .mark_bar(size=24, stroke=pal["surface"], strokeWidth=2)
        .encode(
            x=alt.X("Year:N", sort=years, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("samples:Q", stack="zero", title="Samples"),
            color=class_color(target, labels, pal),
            order=alt.Order("order:Q", sort="ascending"),
            tooltip=[
                alt.Tooltip("Year:N"),
                alt.Tooltip(f"{target}:N", title="SFI class"),
                alt.Tooltip("samples:Q", title="Samples", format=","),
                alt.Tooltip("share:Q", title="Share of year", format=".0%"),
            ],
        )
        .properties(height=300)
    )


def class_cuts(df: pd.DataFrame, target: str, labels: list) -> list:
    """Boundaries between neighbouring classes on the SFI scale."""
    cuts = []
    for lo, hi in zip(labels, labels[1:]):
        a, b = df.loc[df[target] == lo, "SFI"].max(), df.loc[df[target] == hi, "SFI"].min()
        if pd.notna(a) and pd.notna(b):
            cuts.append({"x": (a + b) / 2, "label": f"{lo} | {hi}", "lo_max": a, "hi_min": b})
    return cuts


def chart_sfi_histogram(df: pd.DataFrame, target: str, labels: list, pal: dict, bins: int = 60):
    sfi = df["SFI"].to_numpy(float)
    edges = np.linspace(np.nanmin(sfi), np.nanmax(sfi), bins + 1)
    idx = np.clip(np.digitize(sfi, edges) - 1, 0, bins - 1)
    d = pd.DataFrame({"bin": idx, target: df[target].to_numpy()}).groupby(["bin", target]).size()
    d = d.rename("samples").reset_index()
    d["start"], d["end"] = edges[d["bin"]], edges[d["bin"] + 1]
    d["order"] = d[target].map({c: i for i, c in enumerate(labels)})
    title = "Soil Fertility Index (SFI)"
    bars = (
        alt.Chart(d)
        .mark_bar(stroke=pal["surface"], strokeWidth=2)
        .encode(
            x=alt.X("start:Q", bin="binned", title=title, axis=alt.Axis(format=".2f", tickCount=8)),
            x2="end:Q",
            y=alt.Y("samples:Q", stack="zero", title="Samples"),
            color=class_color(target, labels, pal),
            order=alt.Order("order:Q"),
            tooltip=[
                alt.Tooltip("start:Q", title="SFI from", format=".3f"),
                alt.Tooltip("end:Q", title="SFI to", format=".3f"),
                alt.Tooltip(f"{target}:N", title="SFI class"),
                alt.Tooltip("samples:Q", title="Samples", format=","),
            ],
        )
    )
    cuts = pd.DataFrame(class_cuts(df, target, labels))
    if cuts.empty:
        return bars.properties(height=300)
    rules = alt.Chart(cuts).mark_rule(color=pal["text"], strokeWidth=1).encode(x=alt.X("x:Q", title=title))
    text = (
        alt.Chart(cuts)
        .mark_text(align="left", baseline="top", dx=4, color=pal["text"])
        .encode(x=alt.X("x:Q", title=title), y=alt.value(4), text="label:N")
    )
    return (bars + rules + text).properties(height=300)


def chart_parameter_by_class(df: pd.DataFrame, feature: str, target: str, labels: list, pal: dict, log: bool):
    d = df[[feature, target]].dropna()
    scale = alt.Scale(type="symlog") if log else alt.Scale(zero=False)
    return (
        alt.Chart(d)
        .mark_boxplot(extent=1.5, outliers=False, size=28, rule={"color": pal["text"]}, median={"color": pal["surface"]})
        .encode(
            x=alt.X(f"{target}:N", sort=labels, title="SFI class", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{feature}:Q", scale=scale, title=feature),
            color=class_color(target, labels, pal, legend=False),
        )
        .properties(height=320)
    )


def chart_ranked_bars(d: pd.DataFrame, label: str, value: str, pal: dict, x_title: str, fmt: str):
    """Horizontal bars, largest first, value at the tip. One series -> one colour."""
    order = list(d.sort_values(value, ascending=False)[label])
    base = alt.Chart(d).encode(
        y=alt.Y(f"{label}:N", sort=order, title=None),
        x=alt.X(f"{value}:Q", title=x_title, scale=alt.Scale(domain=[0, float(d[value].max()) * 1.18])),
    )
    bars = base.mark_bar(size=18, cornerRadiusEnd=4, color=pal["series"]).encode(
        tooltip=[alt.Tooltip(f"{label}:N", title="Parameter"), alt.Tooltip(f"{value}:Q", format=fmt)]
    )
    text = base.mark_text(align="left", dx=4, color=pal["text"]).encode(text=alt.Text(f"{value}:Q", format=fmt))
    return (bars + text).properties(height=alt.Step(26))


def heat_ink(frac: float, pal: dict, theme: str) -> str:
    strong = frac > 0.45  # dark cell in light theme, light cell in dark theme
    if theme == "dark":
        return pal["ink_on_light"] if strong else pal["ink_on_dark"]
    return pal["ink_on_dark"] if strong else pal["ink_on_light"]


def chart_confusion(cm: np.ndarray, labels: list, pal: dict, theme: str):
    rows = []
    for i, t in enumerate(labels):
        total = cm[i].sum()
        for j, p in enumerate(labels):
            rows.append({"true": t, "pred": p, "n": int(cm[i, j]), "share": cm[i, j] / total if total else 0.0})
    d = pd.DataFrame(rows)
    vmax = max(int(d["n"].max()), 1)
    d["ink"] = [heat_ink(n / vmax, pal, theme) for n in d["n"]]
    base = alt.Chart(d).encode(
        x=alt.X("pred:N", sort=labels, title="Predicted class", axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("true:N", sort=labels, title="True class"),
    )
    cells = base.mark_rect(stroke=pal["surface"], strokeWidth=2).encode(
        color=alt.Color("n:Q", scale=alt.Scale(domain=[0, vmax], range=pal["heat"]), legend=None),
        tooltip=[
            alt.Tooltip("true:N", title="True"),
            alt.Tooltip("pred:N", title="Predicted"),
            alt.Tooltip("n:Q", title="Samples", format=","),
            alt.Tooltip("share:Q", title="Share of true class", format=".1%"),
        ],
    )
    text = base.mark_text(fontSize=15).encode(text=alt.Text("n:Q", format=","), color=alt.Color("ink:N", scale=None))
    # independent: the text layer uses literal colours (scale=None), which must not merge with the cell scale
    return (cells + text).resolve_scale(color="independent").properties(width=alt.Step(96), height=alt.Step(64))


def chart_shap_importance(values: np.ndarray, features: list, labels: list, pal: dict):
    mean_abs = np.abs(values).mean(axis=0)  # (features, classes)
    d = pd.DataFrame(mean_abs, index=features, columns=labels).rename_axis("feature").reset_index()
    d = d.melt(id_vars="feature", var_name="cls", value_name="mean_abs")
    d["order"] = d["cls"].map({c: i for i, c in enumerate(labels)})
    order = list(pd.Series(mean_abs.sum(axis=1), index=features).sort_values(ascending=False).index)
    return (
        alt.Chart(d)
        .mark_bar(size=18, stroke=pal["surface"], strokeWidth=2)
        .encode(
            y=alt.Y("feature:N", sort=order, title=None),
            x=alt.X("mean_abs:Q", stack="zero", title="Mean |SHAP value| (log-odds), stacked over classes"),
            color=class_color("cls", labels, pal),
            order=alt.Order("order:Q"),
            tooltip=[
                alt.Tooltip("feature:N", title="Parameter"),
                alt.Tooltip("cls:N", title="Class"),
                alt.Tooltip("mean_abs:Q", title="Mean |SHAP|", format=".3f"),
            ],
        )
        .properties(height=alt.Step(26))
    )


def chart_beeswarm(values_k: np.ndarray, X: pd.DataFrame, cls: str, pal: dict):
    """One dot per sample and parameter: x = SHAP value for `cls`, colour = how high the value is."""
    features = list(X.columns)
    order = [features[i] for i in np.argsort(-np.abs(values_k).mean(axis=0))]
    rng = np.random.default_rng(0)
    parts = []
    for j, f in enumerate(features):
        col = X[f]
        parts.append(
            pd.DataFrame(
                {
                    "feature": f,
                    "value": col.to_numpy(),
                    "rank": col.rank(pct=True).to_numpy(),
                    "shap": values_k[:, j],
                    "jitter": rng.uniform(-1, 1, len(col)),
                }
            )
        )
    d = pd.concat(parts, ignore_index=True)
    dots = (
        alt.Chart(d)
        .mark_circle(size=36, opacity=0.6)
        .encode(
            x=alt.X("shap:Q", title=f"SHAP value for {cls} (log-odds):  < pushes away  |  pushes toward >"),
            y=alt.Y("feature:N", sort=order, title=None),
            yOffset=alt.YOffset("jitter:Q", scale=alt.Scale(domain=[-1.25, 1.25])),
            color=alt.Color(
                "rank:Q",
                scale=alt.Scale(domain=[0, 1], range=pal["value"]),
                legend=alt.Legend(title="Parameter value (percentile)", format=".0%", orient="top", gradientLength=180),
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Parameter"),
                alt.Tooltip("value:Q", title="Value", format=".4g"),
                alt.Tooltip("shap:Q", title="SHAP", format="+.3f"),
            ],
        )
    )
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color=pal["baseline"]).encode(x="x:Q")
    return (zero + dots).properties(height=alt.Step(34))


def chart_dependence(x: np.ndarray, s: np.ndarray, feature: str, cls: str, pal: dict, log: bool):
    d = pd.DataFrame({"value": x, "shap": s})
    scale = alt.Scale(type="symlog") if log else alt.Scale(zero=False)
    dots = (
        alt.Chart(d)
        .mark_circle(size=50, opacity=0.5, color=pal["series"])
        .encode(
            x=alt.X("value:Q", scale=scale, title=f"{feature} (measured value)"),
            y=alt.Y("shap:Q", title=f"SHAP value for {cls}"),
            tooltip=[alt.Tooltip("value:Q", title=feature, format=".4g"), alt.Tooltip("shap:Q", title="SHAP", format="+.3f")],
        )
    )
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color=pal["baseline"]).encode(y="y:Q")
    return (zero + dots).properties(height=320)


def chart_probabilities(probs: np.ndarray, labels: list, pal: dict):
    d = pd.DataFrame({"cls": labels, "p": probs})
    base = alt.Chart(d).encode(
        y=alt.Y("cls:N", sort=labels, title=None),
        x=alt.X("p:Q", title="Probability", scale=alt.Scale(domain=[0, 1.18]), axis=alt.Axis(format=".0%", values=[0, 0.25, 0.5, 0.75, 1])),
    )
    bars = base.mark_bar(size=22, cornerRadiusEnd=4).encode(
        color=class_color("cls", labels, pal, legend=False),
        tooltip=[alt.Tooltip("cls:N", title="Class"), alt.Tooltip("p:Q", title="Probability", format=".1%")],
    )
    text = base.mark_text(align="left", dx=4, color=pal["text"]).encode(text=alt.Text("p:Q", format=".1%"))
    return (bars + text).properties(height=alt.Step(40))


def chart_contributions(names: list, contrib: np.ndarray, cls: str, pal: dict):
    toward, away = f"Pushes toward {cls}", f"Pushes away from {cls}"
    d = pd.DataFrame({"label": names, "shap": contrib})
    d["direction"] = np.where(d["shap"] >= 0, toward, away)
    order = list(d.iloc[np.argsort(-np.abs(contrib))]["label"])
    lim = float(np.abs(contrib).max() or 1e-6) * 1.35
    base = alt.Chart(d).encode(
        y=alt.Y("label:N", sort=order, title=None),
        x=alt.X("shap:Q", title=f"SHAP contribution to {cls} (log-odds)", scale=alt.Scale(domain=[-lim, lim])),
    )
    bars = base.mark_bar(size=18).encode(
        color=alt.Color("direction:N", scale=alt.Scale(domain=[toward, away], range=[pal["toward"], pal["away"]]), legend=alt.Legend(title=None, orient="top")),
        tooltip=[alt.Tooltip("label:N", title="Parameter"), alt.Tooltip("shap:Q", title="SHAP", format="+.3f")],
    )
    pos = base.transform_filter("datum.shap >= 0").mark_text(align="left", dx=4, color=pal["text"]).encode(text=alt.Text("shap:Q", format="+.2f"))
    neg = base.transform_filter("datum.shap < 0").mark_text(align="right", dx=-4, color=pal["text"]).encode(text=alt.Text("shap:Q", format="+.2f"))
    zero = alt.Chart(pd.DataFrame({"x": [0.0]})).mark_rule(color=pal["baseline"]).encode(x="x:Q")
    return (zero + bars + pos + neg).properties(height=alt.Step(28))


# --------------------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------------------
def sfi_weights(df: pd.DataFrame):
    """Recover the SFI formula weights from the normalised N_* columns, if they are present."""
    ncols = [f"N_{f}" for f in SOIL_FEATURES]
    if "SFI" not in df.columns or any(c not in df.columns for c in ncols):
        return None
    d = df[ncols + ["SFI"]].dropna()
    A, b = d[ncols].to_numpy(float), d["SFI"].to_numpy(float)
    w, *_ = np.linalg.lstsq(A, b, rcond=None)
    if np.abs(A @ w - b).max() > 1e-3:  # not an exact weighted sum -> don't claim it is
        return None
    return pd.DataFrame({"parameter": SOIL_FEATURES, "weight": np.round(w, 4)})


def tab_overview(data, meta, target, labels, pal):
    if data is None:
        tried = st.session_state.get("_data_tried", [])
        st.info("The dataset overview needs SFI_final_dataset.csv, and it wasn't found. Start the app with its full path:\n\n"
                "`streamlit run app.py -- --data /full/path/to/SFI_final_dataset.csv`\n\n"
                "(The other tabs work without it.)")
        if tried:
            with st.expander("Places checked"):
                st.markdown("\n".join(f"- `{p}`" for p in dict.fromkeys(tried)))
        return
    has_sfi = "SFI" in data.columns and target in data.columns
    unique = len(data.drop_duplicates(subset=[c for c in SOIL_FEATURES + [target] if c in data.columns]))
    c = st.columns(4)
    c[0].metric("Soil samples", f"{len(data):,}")
    c[1].metric("Unique samples", f"{unique:,}", help="Exact duplicates are removed before training.")
    c[2].metric("Survey years", data["Year"].nunique() if "Year" in data.columns else "-")
    c[3].metric("Places", data["Place"].nunique() if "Place" in data.columns else "-")

    if has_sfi:
        cuts = class_cuts(data, target, labels)
        if len(cuts) == 2:
            st.markdown(
                f"**How SFI_Class is defined.** Each sample gets a Soil Fertility Index (SFI) between 0 and 1. "
                f"It is labelled **{labels[0]}** when SFI ≤ {cuts[0]['lo_max']:.4f}, "
                f"**{labels[1]}** from {cuts[0]['hi_min']:.4f} to {cuts[1]['lo_max']:.4f}, "
                f"and **{labels[2]}** from {cuts[1]['hi_min']:.4f}. The model learns this from the raw soil-test values."
            )

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Samples per survey year")
        if "Year" in data.columns:
            st.altair_chart(chart_samples_by_year(data, target, labels, pal))
    with right:
        st.subheader("SFI distribution and class cut-offs")
        if has_sfi:
            st.altair_chart(chart_sfi_histogram(data, target, labels, pal))

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Soil parameters by class")
        params = [f for f in SOIL_FEATURES if f in data.columns]
        f = st.selectbox("Soil parameter", params, index=params.index("OC") if "OC" in params else 0, key="ov_param")
        log = st.checkbox("Log scale (helps with skewed parameters like P, K, S)", key="ov_log")
        st.altair_chart(chart_parameter_by_class(data, f, target, labels, pal, log))
        st.caption("Box = middle 50% of samples, line = median, whiskers = 1.5 × IQR (outliers hidden).")
    with right:
        st.subheader("Weight of each parameter in the SFI")
        w = sfi_weights(data)
        if w is None:
            st.caption("The normalised N_* columns are not in this file, so the SFI weights can't be shown.")
        else:
            st.altair_chart(chart_ranked_bars(w, "parameter", "weight", pal, "Weight in SFI", ".3f"))
            dropped = (meta.get("dropped_redundant_features") or {})
            if dropped:
                notes = ", ".join(f"{a} is an exact copy of {b}" for a, b in dropped.items())
                st.caption(f"Recovered exactly from the data (SFI = Σ weight × normalised score). {notes}, so the model "
                           f"only uses {', '.join(dropped.values())}, which carries both weights.")

    with st.expander("Table view: samples per year and class"):
        if "Year" in data.columns:
            st.dataframe(pd.crosstab(data["Year"], data[target]).reindex(columns=labels).assign(Total=lambda t: t.sum(axis=1)))


def tab_performance(model, meta, art, X_test, y_test, labels, pal, theme):
    m, d, p = meta.get("metrics", {}), meta.get("data", {}), meta.get("params", {})
    features = meta["feature_names"]

    st.subheader("How the model was trained")
    c = st.columns(5)
    c[0].metric("Rows in file", f"{d.get('rows_in_file', 0):,}")
    c[1].metric("After removing duplicates", f"{d.get('rows_used', 0):,}")
    c[2].metric("Training rows", f"{d.get('n_train', len(X_test)):,}")
    c[3].metric("Held-out test rows", f"{d.get('n_test', len(X_test)):,}", help="Never used for training or tuning.")
    c[4].metric("Trees (early stopping)", f"{meta.get('best_iteration', p.get('n_estimators', 0)):,}")
    st.markdown(
        f"**Inputs ({len(features)}):** {', '.join(features)}. Stratified {1 - d.get('test_size', 0.2):.0%}/{d.get('test_size', 0.2):.0%} "
        f"train/test split; 15% of the training rows picked the number of trees by early stopping; "
        f"{len(m.get('cv_fold_accuracies') or [])}-fold cross-validation on the training rows; final model refit on all training rows."
    )

    st.subheader("Results on the held-out test set")
    c = st.columns(5)
    c[0].metric("Accuracy", pct(m.get("test_accuracy")))
    c[1].metric("Balanced accuracy", pct(m.get("test_balanced_accuracy")))
    c[2].metric("Macro F1", pct(m.get("test_macro_f1")))
    c[3].metric("Log loss", f"{m['test_log_loss']:.3f}" if m.get("test_log_loss") is not None else "-", help="Lower is better; rewards confident correct answers.")
    c[4].metric("Low ↔ High mix-ups", m.get("test_low_high_swaps", "-"), help="Errors that skip a whole class. 0 means every mistake is between neighbouring classes.")
    if m.get("cv_accuracy_mean") is not None:
        folds = ", ".join(f"{a:.1%}" for a in m.get("cv_fold_accuracies", []))
        agree = m.get("test_accuracy") is not None and abs(m["test_accuracy"] - m["cv_accuracy_mean"]) < 0.02
        st.caption(f"Cross-validation accuracy {m['cv_accuracy_mean']:.1%} ± {m['cv_accuracy_std']:.1%} (folds: {folds})."
                   + (" Test and CV agree, so the test score isn't a lucky split." if agree else ""))

    proba, pred = score(model, meta, meta.get("trained_at", ""), X_test)
    cm = confusion_matrix(y_test, pred, labels=labels)
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("**Confusion matrix**")
        st.altair_chart(chart_confusion(cm, labels, pal, theme))
    with right:
        st.markdown("**Per-class scores**")
        rep = classification_report(y_test, pred, labels=labels, output_dict=True, zero_division=0)
        table = pd.DataFrame({c: rep[c] for c in labels}).T[["precision", "recall", "f1-score", "support"]]
        table["support"] = table["support"].astype(int)
        st.dataframe(
            table.rename(columns={"precision": "Precision", "recall": "Recall", "f1-score": "F1", "support": "Test samples"}),
            column_config={k: st.column_config.NumberColumn(format="%.3f") for k in ["Precision", "Recall", "F1"]},
        )
        st.caption("Precision: of the samples predicted as this class, how many really are. "
                   "Recall: of the samples that really are this class, how many were found.")

    st.subheader("Which parameters the trees use most")
    imp_path = art / "feature_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        st.altair_chart(chart_ranked_bars(imp, "feature", "gain_pct", pal, "Share of total gain (%)", ".1f"))
        st.caption("LightGBM's own importance (how much each parameter improved the trees). It shows how much, "
                   "not in which direction; the SHAP tab shows both.")

    with st.expander("Hyperparameters, excluded columns and environment"):
        a, b = st.columns(2)
        a.dataframe(pd.DataFrame({"value": {k: str(v) for k, v in p.items()}}))
        excl = meta.get("excluded_columns") or {}
        if excl:
            b.dataframe(pd.DataFrame({"why it is not a model input": excl}))
        st.caption("Versions: " + ", ".join(f"{k} {v}" for k, v in (meta.get("versions") or {}).items())
                   + f" · trained {when(meta.get('trained_at', ''))}")


def tab_shap(model, meta, X_test, labels, pal):
    st.markdown(
        "SHAP splits every prediction into one contribution per soil parameter. For each class, "
        "the model's score = base value + the sum of the contributions; a positive value pushes the sample "
        "**toward** that class, a negative one **away** from it. Values are in log-odds."
    )
    n = min(SHAP_SAMPLE_ROWS, len(X_test))
    Xs = X_test.sample(n=n, random_state=0) if len(X_test) > n else X_test
    values, base = shap_for(model, meta.get("trained_at", ""), Xs)
    features = list(Xs.columns)

    st.subheader("Overall importance")
    st.altair_chart(chart_shap_importance(values, features, labels, pal))
    top = pd.Series(np.abs(values).mean(axis=0).sum(axis=1), index=features).sort_values(ascending=False)
    st.caption(f"Average size of each parameter's push, over {n:,} held-out samples. "
               f"{top.index[0]} matters most, then {top.index[1]} and {top.index[2]}.")

    cls = st.radio("Class to explain", labels, index=len(labels) - 1, horizontal=True, key="shap_class")
    k = labels.index(cls)

    st.subheader(f"How each parameter pushes toward {cls}")
    st.altair_chart(chart_beeswarm(values[:, :, k], Xs, cls, pal))
    st.caption("Each dot is one sample. Dots right of the line push toward the class; dark dots are high values "
               "of that parameter. A clean dark-right / light-left split means 'more of this parameter → more likely this class'.")

    st.subheader("One parameter in detail")
    a, b = st.columns([1, 3])
    with a:
        order = list(top.index)
        f = st.selectbox("Parameter", order, key="dep_feature")
        log = st.checkbox("Log scale", value=True, key="dep_log")
    with b:
        j = features.index(f)
        st.altair_chart(chart_dependence(Xs[f].to_numpy(), values[:, j, k], f, cls, pal, log))
    st.caption(f"Where the dots cross zero is the {f} level at which the model's view flips for {cls}.")

    with st.expander("Table view: mean |SHAP| per parameter and class"):
        t = pd.DataFrame(np.abs(values).mean(axis=0), index=features, columns=labels)
        st.dataframe(t.assign(Total=t.sum(axis=1)).sort_values("Total", ascending=False),
                     column_config={c: st.column_config.NumberColumn(format="%.3f") for c in labels + ["Total"]})


def input_step(col: pd.Series):
    q99 = float(col.quantile(0.99))
    if q99 >= 100:
        return 1.0, "%.1f"
    if q99 >= 10:
        return 0.1, "%.2f"
    return 0.01, "%.3f"


def tab_predict(model, meta, X_train, X_test, y_test, ids, labels, pal):
    features = meta["feature_names"]
    _, test_pred = score(model, meta, meta.get("trained_at", ""), X_test)
    wrong = set(X_test.index[test_pred != y_test.to_numpy()])

    idx = X_test.index
    desc = pd.Series([f"Row {r}" for r in idx], index=idx, dtype=object)
    if ids is not None:
        for col in ids.columns:
            v = ids[col].reindex(idx)
            desc = desc + (" · " + v.astype(str)).where(v.notna(), "")
    desc = desc + " · true " + y_test.astype(str) + np.where(idx.isin(list(wrong)), " (model gets it wrong)", "")

    options = [MEDIAN_OPTION] + desc.tolist()
    row_of = dict(zip(options[1:], idx))
    wrong_opts = [o for o, r in row_of.items() if r in wrong]

    def pick(pool):
        if pool:
            st.session_state["start_choice"] = random.choice(pool)

    if st.session_state.get("start_choice") not in options:
        st.session_state["start_choice"] = MEDIAN_OPTION

    st.markdown("Start from a typical sample or a real held-out test sample, then change any value.")
    a, b, c = st.columns([4, 1, 1])
    with a:
        choice = st.selectbox("Start from", options, key="start_choice")
    with b:
        st.button("Random test sample", on_click=pick, args=(options[1:],), key="rand_any")
    with c:
        st.button("Random mistake", on_click=pick, args=(wrong_opts,), key="rand_wrong", disabled=not wrong_opts,
                  help=f"A test sample the model misclassifies ({len(wrong_opts)} of {len(X_test):,}).")

    rid = row_of.get(choice)
    start = X_train.median() if rid is None else X_test.loc[rid]
    key_part = "median" if rid is None else str(rid)

    values = {}
    cols = st.columns(4)
    lo, hi = X_train.min(), X_train.max()
    for i, f in enumerate(features):
        step, fmt = input_step(X_train[f])
        v = float(start[f])
        with cols[i % 4]:
            values[f] = st.number_input(f, value=v, min_value=min(0.0, v), step=step, format=fmt,
                                        key=f"inp_{f}_{key_part}", help=f"Training data range: {lo[f]:.4g} – {hi[f]:.4g}")
    X1 = pd.DataFrame([values], columns=features).astype("float64")

    outside = [f for f in features if not lo[f] <= values[f] <= hi[f]]
    if outside:
        st.warning("Outside the range seen in training: " + ", ".join(f"{f} ({lo[f]:.4g} – {hi[f]:.4g})" for f in outside)
                   + ". Trees treat such values like the nearest edge of that range, so read this prediction with care.")

    proba, pred = predict(model, meta, X1)
    values_1, base = tree_shap(model, X1)
    p_cls = str(pred[0])

    st.divider()
    left, right = st.columns([2, 3], gap="large")
    with left:
        m1, m2 = st.columns(2)
        m1.metric("Predicted SFI class", p_cls)
        m2.metric("Confidence", f"{proba[0].max():.1%}")
        if rid is not None:
            unchanged = np.allclose(X1.iloc[0].to_numpy(), X_test.loc[rid].to_numpy())
            truth = y_test.loc[rid]
            if not unchanged:
                st.caption(f"Values edited, so the dataset label ({truth}) no longer applies.")
            elif truth == p_cls:
                st.success(f"Correct: the dataset label is {truth}.")
            else:
                st.error(f"Wrong: the dataset label is {truth}.")
        st.altair_chart(chart_probabilities(proba[0], labels, pal))
    with right:
        explain = st.radio("Explain the push toward", labels, index=labels.index(p_cls), horizontal=True, key=f"explain_{key_part}_{p_cls}")
        k = labels.index(explain)
        contrib = values_1[0, :, k]
        names = [f"{f} = {values[f]:.4g}" for f in features]
        st.altair_chart(chart_contributions(names, contrib, explain, pal))
        top = np.argsort(-np.abs(contrib))[:3]
        reasons = "; ".join(f"{features[j]} {'raises' if contrib[j] >= 0 else 'lowers'} it by {abs(contrib[j]):.2f}" for j in top)
        st.caption(f"Score for {explain} = base {base[k]:+.2f} + contributions {contrib.sum():+.2f} = {base[k] + contrib.sum():+.2f}. "
                   f"Biggest effects: {reasons}. The three class scores go through softmax to give the probabilities.")


def tab_batch(model, meta, X_train, X_test, y_test, ids, labels, pal, theme, target):
    features = meta["feature_names"]
    st.markdown(f"Upload a CSV with the columns **{', '.join(features)}** (any order, any case). "
                f"If it also has **{target}**, you'll see how accurate the model is on it.")
    up = st.file_uploader("CSV file", type=["csv"], key="batch_file")
    demo = st.checkbox(f"Use the held-out test set instead ({len(X_test):,} rows)", key="batch_demo")
    if up is None and not demo:
        st.info("Upload a file or tick the box above to try it on the test set.")
        return

    if up is not None:
        df = pd.read_csv(io.BytesIO(up.getvalue()), encoding="utf-8-sig")
        if "Row_ID" in df.columns and df["Row_ID"].is_unique:  # e.g. an X_test.csv / predictions.csv
            df = df.set_index("Row_ID")
        source = up.name
    else:
        df = X_test.copy()
        df[target] = y_test
        if ids is not None:
            df = ids.reindex(df.index).join(df)
        source = "held-out test set"
    df = align_columns(df, features + [target] + ID_COLUMNS)
    if df.index.name != "Row_ID":
        df.index.name = "Row_ID"
    missing = [f for f in features if f not in df.columns]
    if missing:
        st.error(f"{source} is missing these columns: {', '.join(missing)}")
        return

    X = df[features].apply(pd.to_numeric, errors="coerce").astype("float64")
    proba, pred = score(model, meta, meta.get("trained_at", ""), X)
    out = df[[c for c in ID_COLUMNS if c in df.columns]].join(X)
    has_truth = target in df.columns
    if has_truth:
        out[f"{target}_true"] = df[target].astype(str).str.strip()
    out[f"{target}_pred"] = pred
    for j, c in enumerate(labels):
        out[f"prob_{c}"] = proba[:, j]
    out["confidence"] = proba.max(axis=1)
    if has_truth:
        out["correct"] = out[f"{target}_true"] == out[f"{target}_pred"]

    c = st.columns(len(labels) + 2)
    c[0].metric("Rows", f"{len(out):,}")
    for j, cl in enumerate(labels):
        c[j + 1].metric(f"Predicted {cl}", f"{int((pred == cl).sum()):,}")
    if has_truth:
        c[-1].metric("Accuracy", pct(accuracy_score(out[f"{target}_true"], pred)))

    nan_rows = int(X.isna().any(axis=1).sum())
    if nan_rows:
        st.warning(f"{nan_rows:,} row{'s have' if nan_rows != 1 else ' has'} missing or non-numeric values. "
                   "LightGBM still predicts them, but check them first.")
    lo, hi = X_train.min(), X_train.max()
    outside = ((X < lo) | (X > hi)).any(axis=1)
    if outside.any():
        n_out = int(outside.sum())
        st.warning(f"{n_out:,} row{'s have' if n_out != 1 else ' has'} at least one value outside the training range; "
                   "those predictions are less certain.")

    if has_truth:
        left, right = st.columns([1, 2], gap="large")
        with left:
            st.markdown("**Confusion matrix**")
            st.altair_chart(chart_confusion(confusion_matrix(out[f"{target}_true"], pred, labels=labels), labels, pal, theme))
        table_area = right
    else:
        table_area = st.container()
    with table_area:
        st.markdown(f"**Predictions** (first 500 of {len(out):,} rows; download for all)")
        cfg = {f"prob_{c}": st.column_config.NumberColumn(format="%.3f") for c in labels}
        cfg["confidence"] = st.column_config.NumberColumn(format="%.3f")
        st.dataframe(out.head(500), column_config=cfg)

    st.download_button("Download predictions (CSV)", out.to_csv(index_label="Row_ID").encode("utf-8"),
                       file_name="predictions.csv", mime="text/csv", key="dl_pred")

    st.markdown("**SHAP values for these rows**")
    sec = 0.01 * len(X) * meta.get("best_iteration", 1000) / 1000  # measured: ~10 s per 1,000 rows per 1,000 trees on 2 CPU cores
    if st.checkbox(f"Compute SHAP values for all {len(X):,} rows (about {max(sec, 1):.0f} s on 2 CPU cores, faster with more)", key="batch_shap"):
        values, base = shap_for(model, meta.get("trained_at", ""), X)
        cols = {f"shap_{c}_{f}": values[:, j, k] for k, c in enumerate(labels) for j, f in enumerate(features)}
        shap_df = pd.DataFrame(cols, index=X.index)
        for k, c in enumerate(labels):
            shap_df[f"base_{c}"] = base[k]
        st.download_button("Download SHAP values (CSV)", shap_df.to_csv(index_label="Row_ID").encode("utf-8"),
                           file_name="shap_values.csv", mime="text/csv", key="dl_shap")
        st.caption("One column per class and parameter (shap_<class>_<parameter>), plus each class's base value. "
                   "For every row and class: base + sum of that class's columns = the model's raw score.")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
ARGS = parse_cli()


def main() -> None:
    st.set_page_config(page_title="SFI classifier", page_icon="🌱", layout="wide")
    art, checked = find_artifacts(ARGS.artifacts)
    saved = st.session_state.get("artifacts_dir")
    if art is None and saved and (clean_path(saved) / "model_metadata.json").exists():
        art = clean_path(saved)
    if art is None:
        st.error("Can't find the trained model (artifacts/model_metadata.json). Looked in:\n\n"
                 + "\n".join(f"- `{c}`" for c in checked))
        st.markdown(
            "**Fix:** put `app.py` in the folder where you ran `python train.py` (the folder that contains "
            "`artifacts` and `outputs`), open a terminal there and run `streamlit run app.py`. "
            "If you haven't trained yet, run `python train.py` first. Or paste the folder's full path here:"
        )
        st.text_input("Full path of your artifacts folder", key="artifacts_path_input", on_change=_remember_artifacts_path,
                      placeholder=r"C:\Users\you\SFI_project\artifacts")
        if saved:
            st.warning(f"There's no model_metadata.json in `{clean_path(saved)}`.")
        st.stop()
    art = art.resolve()
    meta_path = art / "model_metadata.json"

    model, meta = load_model(str(art), stamp(meta_path))
    features = tuple(meta["feature_names"])
    target = meta.get("target", "SFI_Class")
    labels = class_labels(model, meta)
    X_train, _ = load_split(str(art), "train", features, stamp(art / "X_train.csv"))
    X_test, y_test = load_split(str(art), "test", features, stamp(art / "X_test.csv"))

    data_path, data_tried = find_dataset(ARGS.data, "SFI_final_dataset.csv", meta, art)
    data = load_dataset(str(data_path), stamp(data_path)) if data_path is not None else None
    if data is None:
        st.session_state["_data_tried"] = [str(p) for p in data_tried]
    if data is not None:
        data = align_columns(data, SOIL_FEATURES + [target] + ID_COLUMNS)
    same_file = data is not None and file_sha256(str(data_path), stamp(data_path)) == meta.get("data", {}).get("sha256")
    ids = data[[c for c in ["Year", "Place"] if c in data.columns]] if same_file else None

    theme = ui_theme()
    pal = PALETTES[theme]
    m = meta.get("metrics", {})

    with st.sidebar:
        st.markdown("### Model")
        st.markdown(f"LightGBM {meta.get('versions', {}).get('lightgbm', '')}  \n"
                    f"{meta.get('best_iteration', 0):,} trees · {len(features)} inputs  \n"
                    f"Test accuracy **{pct(m.get('test_accuracy'))}**")
        st.caption(f"Trained {when(meta.get('trained_at', ''))}")
        st.caption(f"Artifacts: `{art}`  \nDataset: `{data_path if data is not None else 'not found'}`")
        st.divider()
        st.markdown("**Showcase:** Overview, Model performance, SHAP explanations  \n"
                    "**Testing:** Predict a sample, Batch prediction")

    st.title("Soil Fertility Index classifier")
    st.caption(f"LightGBM model that predicts {target} ({' / '.join(labels)}) from {len(features)} soil-test parameters.")

    tabs = st.tabs(["Overview", "Model performance", "SHAP explanations", "Predict a sample", "Batch prediction"])
    with tabs[0]:
        tab_overview(data, meta, target, labels, pal)
    with tabs[1]:
        tab_performance(model, meta, art, X_test, y_test, labels, pal, theme)
    with tabs[2]:
        tab_shap(model, meta, X_test, labels, pal)
    with tabs[3]:
        tab_predict(model, meta, X_train, X_test, y_test, ids, labels, pal)
    with tabs[4]:
        tab_batch(model, meta, X_train, X_test, y_test, ids, labels, pal, theme, target)


if __name__ == "__main__":
    main()