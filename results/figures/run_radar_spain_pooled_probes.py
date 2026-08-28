"""Linear probes on pooled Spanish RADAR sites (CIBER + IISPV)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parents[2]
EMBED_DIR = BASE / "data" / "processed" / "Nick" / "Nick_repr_learn"
RESULTS = BASE / "results" / "metrics" / "repr_learn"
N_FOLDS = 5
ENCODERS = ("wav2vec2", "hubert", "whisper")
SPAIN_SITES = ("CIBER", "IISPV")


def load_site_embeddings(site: str, encoder: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = EMBED_DIR / f"radar_{site}_{encoder}_embeddings.csv"
    df = pd.read_csv(path)
    df["participant_id"] = df["participant_id"].astype(str).str.strip()
    if "label" in df.columns:
        df["depressed"] = pd.to_numeric(df["label"], errors="coerce")
    else:
        df["depressed"] = (pd.to_numeric(df["phq8_score"], errors="coerce") >= 10).astype(float)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    df = df.dropna(subset=["participant_id", "depressed"] + emb_cols).copy()
    df["depressed"] = df["depressed"].astype(int)
    groups = np.array([f"{site}_{g}" for g in df["participant_id"].astype(str)], dtype=str)
    return df[emb_cols].to_numpy(dtype=np.float32), df["depressed"].to_numpy(dtype=int), groups


def run_group_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_rows: list[dict] = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups), start=1):
        pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        solver="lbfgs",
                        random_state=42,
                    ),
                ),
            ]
        )
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        proba = pipe.predict_proba(X[te])[:, 1]
        dummy = DummyClassifier(strategy="stratified", random_state=42)
        dummy.fit(X[tr], y[tr])
        dummy_pred = dummy.predict(X[te])
        err_m = (pred != y[te]).astype(float)
        err_d = (dummy_pred != y[te]).astype(float)
        if len(err_m) >= 2 and not np.allclose(err_m, err_d):
            w_p = float(wilcoxon(err_m, err_d, zero_method="wilcox", mode="auto").pvalue)
        else:
            w_p = float("nan")
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "accuracy": accuracy_score(y[te], pred),
                "f1": f1_score(y[te], pred),
                "roc_auc": roc_auc_score(y[te], proba),
                "wilcoxon_p_vs_stratified_dummy": w_p,
            }
        )
    folds_df = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(
        [
            {
                "subset": "all",
                "n_rows": len(y),
                "n_groups": len(np.unique(groups)),
                "accuracy_mean": folds_df["accuracy"].mean(),
                "accuracy_std": folds_df["accuracy"].std(),
                "f1_mean": folds_df["f1"].mean(),
                "f1_std": folds_df["f1"].std(),
                "roc_auc_mean": folds_df["roc_auc"].mean(),
                "roc_auc_std": folds_df["roc_auc"].std(),
            }
        ]
    )
    return folds_df, summary


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    for encoder in ENCODERS:
        xs, ys, gs = [], [], []
        for site in SPAIN_SITES:
            X, y, groups = load_site_embeddings(site, encoder)
            print(f"{encoder} {site}: {len(y)} rows, {len(np.unique(groups))} groups")
            xs.append(X)
            ys.append(y)
            gs.append(groups)
        X = np.vstack(xs)
        y = np.concatenate(ys)
        groups = np.concatenate(gs)
        print(f"{encoder} pooled Spain: {len(y)} rows, {len(np.unique(groups))} groups")
        folds_df, summary_df = run_group_cv(X, y, groups)
        stem = f"radar_CIBER_IISPV_{encoder}"
        folds_df.to_csv(RESULTS / f"{stem}_cv_folds.csv", index=False)
        summary_df.to_csv(RESULTS / f"{stem}_summary.csv", index=False)
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
