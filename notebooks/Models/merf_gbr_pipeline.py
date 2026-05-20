"""MERF with Gradient Boosting Regressor + full evaluation (mirrors Gradient_Boosting copy GBR)."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import kruskal, wilcoxon
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from merf.merf import MERF

try:
    from aif360.datasets import BinaryLabelDataset
    from aif360.metrics import BinaryLabelDatasetMetric
except ImportError:
    BinaryLabelDataset = None
    BinaryLabelDatasetMetric = None


BASE = Path(r"C:/Users/janku/Documents/KCL/Research Project/Research Project")
EXCEL_EXPORT_PATH = BASE / "results/logs/merf_gbr_results.xlsx"

RESULTS_STORE = {
    "split_info": [],
    "best_params": [],
    "train_test_balance": [],
    "cv_folds": [],
    "cv_balance": [],
    "statistical_tests": [],
    "cv_summary": [],
    "test_summary": [],
    "subgroup_gender": [],
    "subgroup_age": [],
    "subgroup_gender_age": [],
    "fairness": [],
    "wilcoxon": [],
    "confusion_matrices": [],
}


def reset_results_store():
    for key in RESULTS_STORE:
        RESULTS_STORE[key] = []


def record_table(sheet_key, model_name, df):
    if df is None or len(df) == 0:
        return
    out = df.copy() if not isinstance(df, pd.Series) else df.to_frame().T
    if "model" not in out.columns:
        out.insert(0, "model", model_name)
    RESULTS_STORE.setdefault(sheet_key, []).append(out)


def record_confusion_matrix(model_name, cm, matrix_name):
    cm = np.asarray(cm)
    rows = []
    for i, true_lab in enumerate(["control_0", "depressed_1"]):
        for j, pred_lab in enumerate(["control_0", "depressed_1"]):
            rows.append({
                "model": model_name,
                "matrix": matrix_name,
                "true_label": true_lab,
                "pred_label": pred_lab,
                "count": int(cm[i, j]),
            })
    RESULTS_STORE.setdefault("confusion_matrices", []).append(pd.DataFrame(rows))


def record_standard_model_outputs(model_name, **kwargs):
    mapping = {
        "split_info": "split_info",
        "best_params": "best_params",
        "train_test_balance": "train_test_balance",
        "cv_folds": "cv_folds",
        "cv_balance": "cv_balance",
        "statistical_tests": "statistical_tests",
        "cv_summary": "cv_summary",
        "test_summary": "test_summary",
        "wilcoxon": "wilcoxon",
        "confusion_matrices": None,
    }
    for key, sheet in mapping.items():
        val = kwargs.get(key)
        if val is None:
            continue
        if key == "confusion_matrices":
            for matrix_name, cm in val.items():
                record_confusion_matrix(model_name, cm, matrix_name)
        elif key == "wilcoxon":
            for test_name, p_value in val.items():
                record_table("wilcoxon", model_name, pd.DataFrame([{"test_name": test_name, "p_value": p_value}]))
        elif key == "split_info":
            record_table(sheet, model_name, pd.DataFrame([val]))
        elif key == "best_params":
            record_table(sheet, model_name, pd.DataFrame([val]))
        else:
            record_table(sheet, model_name, val)


def export_results_to_excel(path=EXCEL_EXPORT_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        wrote = False
        for sheet_key, frames in RESULTS_STORE.items():
            if not frames:
                continue
            pd.concat(frames, ignore_index=True).to_excel(writer, sheet_name=sheet_key[:31], index=False)
            wrote = True
        if not wrote:
            pd.DataFrame({"message": ["No results recorded."]}).to_excel(writer, sheet_name="info", index=False)
    print(f"\nExcel workbook saved to: {path.resolve()}")
    return path


def confusion_matrix_for_display(y_true, y_pred, title):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["control (0)", "depressed (1)"]).plot(ax=ax, colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()
    plt.close(fig)
    return cm


def _safe_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def evaluate_by_group(y_true, y_pred, y_score, sensitive_feature):
    eval_df = pd.DataFrame({
        "y_true": np.asarray(y_true).astype(int),
        "y_pred": np.asarray(y_pred).astype(int),
        "y_score": np.asarray(y_score, dtype=float),
        "group": pd.Series(sensitive_feature).reset_index(drop=True),
    }).dropna(subset=["group"])
    rows = []
    for group_value, group_df in eval_df.groupby("group", dropna=True):
        rows.append({
            "group": group_value,
            "n": len(group_df),
            "n_depressed": int(group_df["y_true"].sum()),
            "n_control": int((group_df["y_true"] == 0).sum()),
            "accuracy": accuracy_score(group_df["y_true"], group_df["y_pred"]),
            "f1": f1_score(group_df["y_true"], group_df["y_pred"], zero_division=0),
            "roc_auc": _safe_auc(group_df["y_true"], group_df["y_score"]),
        })
    return pd.DataFrame(rows)


def _gender_from_file_stem(file_stem):
    extracted = pd.Series(file_stem).astype(str).str.extract(r"^[0-9]+_[A-Z]?([FM])\d+", expand=False)
    return extracted.map({"F": "female", "M": "male"})


def _age_from_file_stem(file_stem):
    return pd.to_numeric(
        pd.Series(file_stem).astype(str).str.extract(r"^[0-9]+_[A-Z]?[FM](\d+)", expand=False),
        errors="coerce",
    )


def _format_gender(gender_values):
    gender = pd.Series(gender_values).reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(gender):
        return "gender_" + gender.astype("Int64").astype(str)
    normalized = gender.astype(str).str.strip().str.lower()
    return normalized.replace({"f": "female", "female": "female", "m": "male", "male": "male"})


def make_subgroup_frame(df_rows):
    rows = df_rows.reset_index(drop=True)
    subgroup_df = pd.DataFrame(index=rows.index)
    if "Gender" in rows.columns:
        subgroup_df["gender"] = _format_gender(rows["Gender"])
    elif "file_stem" in rows.columns:
        subgroup_df["gender"] = _gender_from_file_stem(rows["file_stem"])
    if "Age" in rows.columns:
        age = pd.to_numeric(rows["Age"], errors="coerce")
    elif "file_stem" in rows.columns:
        age = _age_from_file_stem(rows["file_stem"])
    else:
        age = pd.Series(np.nan, index=rows.index)
    subgroup_df["age_group"] = pd.cut(age, bins=[-np.inf, 35, 55, np.inf], labels=["young", "middle", "older"]).astype("object")
    if {"gender", "age_group"}.issubset(subgroup_df.columns):
        subgroup_df["gender_age_group"] = (
            subgroup_df["age_group"].astype(str) + " " + subgroup_df["gender"].astype(str)
        )
        subgroup_df.loc[subgroup_df["gender"].isna() | subgroup_df["age_group"].isna(), "gender_age_group"] = np.nan
    return subgroup_df


def _gender_binary_code(df_rows):
    rows = df_rows.reset_index(drop=True)
    if "Gender" in rows.columns:
        return pd.to_numeric(rows["Gender"], errors="coerce")
    if "file_stem" in rows.columns:
        return _gender_from_file_stem(rows["file_stem"]).map({"female": 0, "male": 1})
    return pd.Series(np.nan, index=rows.index)


def print_aif360_fairness_metrics(model_name, y_pred, df_test_rows):
    if BinaryLabelDataset is None:
        print(f"\n{model_name} AIF360 skipped (package not installed)")
        return
    gender = _gender_binary_code(df_test_rows)
    valid = gender.notna()
    if valid.sum() < 2 or len(np.unique(gender[valid])) < 2:
        print(f"\n{model_name} AIF360 fairness metrics: insufficient gender groups")
        return
    fairness_df = pd.DataFrame({
        "gender": gender[valid].astype(int).to_numpy(),
        "depressed": np.asarray(y_pred, dtype=int).reshape(-1, 1)[valid.to_numpy()].ravel(),
        "dummy": 1,
    })
    dataset = BinaryLabelDataset(
        df=fairness_df,
        label_names=["depressed"],
        protected_attribute_names=["gender"],
        favorable_label=1,
        unfavorable_label=0,
    )
    metric = BinaryLabelDatasetMetric(
        dataset,
        privileged_groups=[{"gender": 1}],
        unprivileged_groups=[{"gender": 0}],
    )
    out = pd.DataFrame([{
        "disparate_impact": float(metric.disparate_impact()),
        "statistical_parity_difference": float(metric.statistical_parity_difference()),
    }])
    record_table("fairness", model_name, out)
    print(f"\n{model_name} AIF360 fairness metrics (held-out predictions):")
    print(out.to_string(index=False))


def print_subgroup_results(model_name, y_true, y_pred, y_score, subgroup_df, df_test_rows):
    sheet_map = {"gender": "subgroup_gender", "age_group": "subgroup_age", "gender_age_group": "subgroup_gender_age"}
    for column, label in [("gender", "Gender"), ("age_group", "Age group"), ("gender_age_group", "Gender x age group")]:
        if column not in subgroup_df.columns:
            print(f"\n{model_name} {label} results: unavailable")
            continue
        results = evaluate_by_group(y_true, y_pred, y_score, subgroup_df[column])
        print(f"\n{model_name} {label} results:")
        print(results)
        record_table(sheet_map[column], model_name, results)
    print_aif360_fairness_metrics(model_name, y_pred, df_test_rows)


def run_shap_lime_explanations(model_name, model, X_train, X_test, feature_names, mode="regression"):
    try:
        import shap
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        print(f"\n{model_name} SHAP/LIME skipped (install shap and lime)")
        return
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X_train), size=min(1000, len(X_train)), replace=False)
    test_idx = rng.choice(len(X_test), size=min(500, len(X_test)), replace=False)
    X_bg, X_shap = X_train[bg_idx], X_test[test_idx]
    print(f"\n{model_name} SHAP summary (n={len(X_shap)}):")
    try:
        explainer = shap.Explainer(model, X_bg)
        shap_values = explainer(X_shap)
        shap.summary_plot(shap_values, pd.DataFrame(X_shap, columns=feature_names), show=False)
        plt.tight_layout()
        plt.show()
        plt.close("all")
    except Exception as exc:
        print(f"  SHAP failed: {exc}")
    predict_fn = model.predict if mode == "regression" else model.predict_proba
    print(f"\n{model_name} LIME (2 instances):")
    try:
        lime_explainer = LimeTabularExplainer(X_bg, feature_names=list(feature_names), mode=mode, discretize_continuous=True)
        for i in range(min(2, len(X_test))):
            exp = lime_explainer.explain_instance(X_test[i], predict_fn, num_features=min(10, len(feature_names)))
            fig = exp.as_pyplot_figure()
            plt.tight_layout()
            plt.show()
            plt.close(fig)
    except Exception as exc:
        print(f"  LIME failed: {exc}")


def _bdi_threshold_from_train(y_train, y_bin_train, default=13.0):
    """Threshold between control/depressed BDI on training rows (midpoint)."""
    ctrl = y_train[y_bin_train == 0]
    dep = y_train[y_bin_train == 1]
    if len(ctrl) and len(dep):
        return float((np.max(ctrl) + np.min(dep)) / 2.0)
    return default


def _wilcoxon_paired(y_true, pred_a, pred_b):
    e1 = np.abs(np.asarray(y_true, dtype=float) - np.asarray(pred_a, dtype=float))
    e2 = np.abs(np.asarray(y_true, dtype=float) - np.asarray(pred_b, dtype=float))
    if len(e1) < 2 or np.allclose(e1, e2):
        return float("nan")
    return float(wilcoxon(e1, e2, zero_method="wilcox", mode="auto").pvalue)


def _as_cluster_series(clusters):
    return pd.Series(np.asarray(clusters).ravel(), name="cluster")


def _fit_merf_predict(X_train, Z_train, clusters_train, y_train, X_test, Z_test, clusters_test, gbr_params):
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    Z_train_a = np.asarray(Z_train, dtype=float)
    Z_test_a = np.asarray(Z_test, dtype=float)
    clusters_train_s = _as_cluster_series(clusters_train)
    clusters_test_s = _as_cluster_series(clusters_test)
    gbr = GradientBoostingRegressor(**gbr_params, random_state=42)
    merf_model = MERF(fixed_effects_model=gbr, max_iterations=20)
    merf_model.fit(X_train_s, Z_train_a, clusters_train_s, y_train)
    y_pred = merf_model.predict(X_test_s, Z_test_a, clusters_test_s)
    return y_pred, scaler, gbr, merf_model


def run_merf_gbr_radar():
    model_name = "RADAR MERF-GBR"
    print("\n" + "=" * 80)
    print(f"MODEL SECTION: {model_name}")
    print("=" * 80 + "\n")

    dataset = BASE / "data/processed/radar_model_dataset_raw_features.csv"
    df = pd.read_csv(dataset)
    df["participant_id"] = df["participant_id"].astype(str).str.strip()

    x_candidates = [
        "Speaking_Rate", "Articulation_Rate", "Phonation_Ratio", "Pause_Rate", "Pause_Ratio",
        "mean_F0", "stdev_F0_Semitone", "HNR_dB", "Spectral_Slope", "Spectral_Tilt",
        "Cepstral_Peak_Prominence", "mean_F1_Loc", "std_F1_Loc", "mean_B1_Loc", "std_B1_Loc",
        "mean_F2_Loc", "std_F2_Loc", "mean_B2_Loc", "std_B2_Loc", "Spectral_Gravity", "Spectral_Std_Dev",
    ]
    z_features = [c for c in ["Age", "Gender", "Education_Years", "Height"] if c in df.columns]
    feature_cols = [c for c in x_candidates if c in df.columns]
    required = feature_cols + z_features + ["phq8_score", "participant_id"]
    df = df.dropna(subset=required).copy()

    df["depressed"] = (df["phq8_score"] >= 10).astype(int)
    X = df[feature_cols].values
    y = df["phq8_score"].values
    y_bin = df["depressed"].values
    groups = df["participant_id"].values
    Z = df[z_features].copy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    y_bin_train, y_bin_test = y_bin[train_idx], y_bin[test_idx]
    Z_train, Z_test = Z.iloc[train_idx], Z.iloc[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]

    print("Train rows:", len(train_idx), "| Test rows:", len(test_idx))
    print("Train participants:", len(np.unique(groups_train)), "| Test:", len(np.unique(groups_test)))

    train_test_stat, train_test_p = kruskal(y_train, y_test)
    train_test_balance_df = pd.DataFrame([{
        "comparison": "train_vs_test",
        "train_n": len(y_train),
        "test_n": len(y_test),
        "train_mean_phq8": float(np.mean(y_train)),
        "test_mean_phq8": float(np.mean(y_test)),
        "kruskal_H": train_test_stat,
        "p_value": train_test_p,
    }])
    print("\nTrain/Test PHQ-8 balance:")
    print(train_test_balance_df)

    gkf = GroupKFold(n_splits=5)
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.03, 0.05],
        "max_depth": [2, 3],
    }
    grid = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        scoring="neg_root_mean_squared_error",
        cv=gkf,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train, groups=groups_train)
    best_gbr_params = grid.best_params_
    print("Best GBR params from GridSearchCV:", best_gbr_params)

    cv_results = []
    cv_fold_phq8 = []
    cv_y_true_bin, cv_y_pred_bin = [], []

    for fold, (tr, va) in enumerate(gkf.split(X_train, y_train, groups=groups_train), start=1):
        y_va = y_train[va]
        y_pred, _, _, _ = _fit_merf_predict(
            X_train[tr], Z.iloc[tr], groups_train[tr], y_train[tr],
            X_train[va], Z.iloc[va], groups_train[va], y_va,
            best_gbr_params,
        )
        preds_bin = (y_pred >= 10).astype(int)
        cv_y_true_bin.extend(y_bin_train[va])
        cv_y_pred_bin.extend(preds_bin)
        cv_fold_phq8.append(y_va)

        mean_train = float(np.mean(y_train[tr]))
        w_cv = _wilcoxon_paired(y_va, y_pred, np.full_like(y_va, mean_train))
        print(f"  Wilcoxon p (|y-ŷ| MERF-GBR vs |y-ȳ_train|) fold {fold}:", w_cv)

        cv_results.append({
            "fold": fold,
            "mae": mean_absolute_error(y_va, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_va, y_pred)),
            "r2": r2_score(y_va, y_pred),
            "accuracy": accuracy_score(y_bin_train[va], preds_bin),
            "f1": f1_score(y_bin_train[va], preds_bin),
            "roc_auc": roc_auc_score(y_bin_train[va], y_pred) if len(np.unique(y_bin_train[va])) > 1 else np.nan,
            "wilcoxon_p_vs_train_mean_baseline": w_cv,
        })

    cv_results_df = pd.DataFrame(cv_results)
    print("\nCV fold results:")
    print(cv_results_df)

    cv_balance_rows = [
        {"fold": i, "n_rows": len(v), "mean_phq8": float(np.mean(v)), "median_phq8": float(np.median(v))}
        for i, v in enumerate(cv_fold_phq8, start=1)
    ]
    cv_balance_df = pd.DataFrame(cv_balance_rows)
    cv_stat, cv_p = kruskal(*cv_fold_phq8)
    cv_balance_summary_df = pd.DataFrame([{"comparison": "cv_folds", "kruskal_H": cv_stat, "p_value": cv_p}])

    cv_summary_df = pd.DataFrame([{
        "subset": "cv_train",
        "n_rows": len(train_idx),
        "n_depressed": int(y_bin_train.sum()),
        "n_control": int((y_bin_train == 0).sum()),
        "mae_mean": cv_results_df["mae"].mean(),
        "mae_std": cv_results_df["mae"].std(),
        "rmse_mean": cv_results_df["rmse"].mean(),
        "rmse_std": cv_results_df["rmse"].std(),
        "r2_mean": cv_results_df["r2"].mean(),
        "r2_std": cv_results_df["r2"].std(),
        "accuracy_mean": cv_results_df["accuracy"].mean(),
        "accuracy_std": cv_results_df["accuracy"].std(),
        "f1_mean": cv_results_df["f1"].mean(),
        "f1_std": cv_results_df["f1"].std(),
        "roc_auc_mean": cv_results_df["roc_auc"].mean(),
        "roc_auc_std": cv_results_df["roc_auc"].std(),
    }])
    print("\nCV summary:")
    print(cv_summary_df)

    test_preds, scaler, gbr_explain, _ = _fit_merf_predict(
        X_train, Z_train, groups_train, y_train,
        X_test, Z_test, groups_test, y_test,
        best_gbr_params,
    )
    test_preds_bin = (test_preds >= 10).astype(int)

    test_summary_df = pd.DataFrame([{
        "subset": "held_out_test",
        "n_rows": len(test_idx),
        "n_depressed": int(y_bin_test.sum()),
        "n_control": int((y_bin_test == 0).sum()),
        "mae": mean_absolute_error(y_test, test_preds),
        "rmse": np.sqrt(mean_squared_error(y_test, test_preds)),
        "r2": r2_score(y_test, test_preds),
        "accuracy": accuracy_score(y_bin_test, test_preds_bin),
        "f1": f1_score(y_bin_test, test_preds_bin),
        "roc_auc": roc_auc_score(y_bin_test, test_preds),
    }])
    print("\nHeld-out test results:")
    print(test_summary_df)

    w_test = _wilcoxon_paired(y_test, test_preds, np.full_like(y_test, float(np.mean(y_train))))
    print("\nWilcoxon p (held-out, |y-ŷ| MERF-GBR vs |y-ȳ_train|):", w_test)

    test_subgroups = make_subgroup_frame(df.iloc[test_idx])
    print_subgroup_results(model_name, y_bin_test, test_preds_bin, test_preds, test_subgroups, df.iloc[test_idx])

    cm_cv = confusion_matrix_for_display(cv_y_true_bin, cv_y_pred_bin, title=f"{model_name} — pooled CV")
    cm_test = confusion_matrix_for_display(y_bin_test, test_preds_bin, title=f"{model_name} — held-out test")
    print("\nPooled CV confusion matrix:\n", cm_cv)
    print("\nHeld-out test confusion matrix:\n", cm_test)

    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    gbr_explain.fit(X_train_s, y_train)
    run_shap_lime_explanations(model_name, gbr_explain, X_train_s, X_test_s, feature_cols, mode="regression")

    record_standard_model_outputs(
        model_name,
        split_info={
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "train_participants": len(np.unique(groups_train)),
            "test_participants": len(np.unique(groups_test)),
        },
        best_params={"model": model_name, **best_gbr_params},
        train_test_balance=train_test_balance_df,
        cv_folds=cv_results_df,
        cv_balance=cv_balance_df,
        statistical_tests=cv_balance_summary_df,
        cv_summary=cv_summary_df,
        test_summary=test_summary_df,
        wilcoxon={"held_out_merf_gbr_vs_train_mean": w_test},
        confusion_matrices={"cv_pooled": cm_cv, "held_out_test": cm_test},
    )


def run_merf_gbr_androids():
    model_name = "Androids MERF-GBR"
    print("\n" + "=" * 80)
    print(f"MODEL SECTION: {model_name}")
    print("=" * 80 + "\n")

    dataset = BASE / "data/processed/androids_model_dataset_basic.csv"
    df = pd.read_csv(dataset)
    df["file_stem"] = df["file_stem"].astype(str).str.strip()

    meta_cols = ["file_path", "file", "file_stem", "bdi_score", "depressed", "fold", "speech_type", "subgroup_from_path"]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    z_features = [c for c in ["speech_type", "subgroup_from_path"] if c in df.columns]
    required = feature_cols + ["bdi_score", "depressed", "file_stem"] + z_features
    df = df.dropna(subset=required).copy()

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    X = df[feature_cols].astype(float).values
    y = df["bdi_score"].astype(float).values
    y_bin = df["depressed"].astype(int).values
    groups = df["file_stem"].values
    Z = pd.get_dummies(df[z_features], drop_first=True).astype(float)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    y_bin_train, y_bin_test = y_bin[train_idx], y_bin[test_idx]
    Z_train, Z_test = Z.iloc[train_idx], Z.iloc[test_idx]
    groups_train, groups_test = groups[train_idx], groups[test_idx]

    print("Train rows:", len(train_idx), "| Test rows:", len(test_idx))
    print("Groups train:", len(np.unique(groups_train)), "| test:", len(np.unique(groups_test)))

    train_test_stat, train_test_p = kruskal(y_train, y_test)
    train_test_balance_df = pd.DataFrame([{
        "comparison": "train_vs_test",
        "train_n": len(y_train),
        "test_n": len(y_test),
        "train_mean_bdi": float(np.mean(y_train)),
        "test_mean_bdi": float(np.mean(y_test)),
        "kruskal_H": train_test_stat,
        "p_value": train_test_p,
    }])
    print("\nTrain/Test BDI balance:")
    print(train_test_balance_df)

    gkf = GroupKFold(n_splits=5)
    param_grid = {"n_estimators": [100, 200], "learning_rate": [0.03, 0.05], "max_depth": [2, 3]}
    grid = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        scoring="neg_root_mean_squared_error",
        cv=gkf,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train, groups=groups_train)
    best_gbr_params = grid.best_params_
    print("Best GBR params from GridSearchCV:", best_gbr_params)

    bdi_thresh = _bdi_threshold_from_train(y_train, y_bin_train)
    print(f"BDI threshold for binary metrics (train midpoint): {bdi_thresh:.2f}")

    cv_results = []
    cv_y_true_bin, cv_y_pred_bin = [], []

    for fold, (tr, va) in enumerate(gkf.split(X_train, y_train, groups=groups_train), start=1):
        y_va = y_train[va]
        y_pred, _, _, _ = _fit_merf_predict(
            X_train[tr], Z.iloc[tr], groups_train[tr], y_train[tr],
            X_train[va], Z.iloc[va], groups_train[va], y_va,
            best_gbr_params,
        )
        preds_bin = (y_pred >= bdi_thresh).astype(int)
        cv_y_true_bin.extend(y_bin_train[va])
        cv_y_pred_bin.extend(preds_bin)
        mean_train = float(np.mean(y_train[tr]))
        w_cv = _wilcoxon_paired(y_va, y_pred, np.full_like(y_va, mean_train))
        print(f"  Wilcoxon p fold {fold}:", w_cv)
        cv_results.append({
            "fold": fold,
            "mae": mean_absolute_error(y_va, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_va, y_pred)),
            "r2": r2_score(y_va, y_pred),
            "accuracy": accuracy_score(y_bin_train[va], preds_bin),
            "f1": f1_score(y_bin_train[va], preds_bin),
            "roc_auc": roc_auc_score(y_bin_train[va], y_pred) if len(np.unique(y_bin_train[va])) > 1 else np.nan,
            "wilcoxon_p_vs_train_mean_baseline": w_cv,
        })

    cv_results_df = pd.DataFrame(cv_results)
    print("\nCV fold results:")
    print(cv_results_df)

    cv_summary_df = pd.DataFrame([{
        "subset": "cv_train",
        "n_rows": len(train_idx),
        "n_depressed": int(y_bin_train.sum()),
        "n_control": int((y_bin_train == 0).sum()),
        "mae_mean": cv_results_df["mae"].mean(),
        "rmse_mean": cv_results_df["rmse"].mean(),
        "r2_mean": cv_results_df["r2"].mean(),
        "accuracy_mean": cv_results_df["accuracy"].mean(),
        "f1_mean": cv_results_df["f1"].mean(),
        "roc_auc_mean": cv_results_df["roc_auc"].mean(),
    }])
    print("\nCV summary:")
    print(cv_summary_df)

    test_preds, scaler, gbr_explain, _ = _fit_merf_predict(
        X_train, Z_train, groups_train, y_train,
        X_test, Z_test, groups_test, y_test,
        best_gbr_params,
    )
    test_preds_bin = (test_preds >= bdi_thresh).astype(int)

    test_summary_df = pd.DataFrame([{
        "subset": "held_out_test",
        "n_rows": len(test_idx),
        "n_depressed": int(y_bin_test.sum()),
        "n_control": int((y_bin_test == 0).sum()),
        "mae": mean_absolute_error(y_test, test_preds),
        "rmse": np.sqrt(mean_squared_error(y_test, test_preds)),
        "r2": r2_score(y_test, test_preds),
        "accuracy": accuracy_score(y_bin_test, test_preds_bin),
        "f1": f1_score(y_bin_test, test_preds_bin),
        "roc_auc": roc_auc_score(y_bin_test, test_preds) if len(np.unique(y_bin_test)) > 1 else np.nan,
        "bdi_threshold": bdi_thresh,
    }])
    print("\nHeld-out test results:")
    print(test_summary_df)

    w_test = _wilcoxon_paired(y_test, test_preds, np.full_like(y_test, float(np.mean(y_train))))
    print("\nWilcoxon p (held-out):", w_test)

    test_subgroups = make_subgroup_frame(df.iloc[test_idx])
    print_subgroup_results(model_name, y_bin_test, test_preds_bin, test_preds, test_subgroups, df.iloc[test_idx])

    cm_cv = confusion_matrix_for_display(cv_y_true_bin, cv_y_pred_bin, title=f"{model_name} — pooled CV")
    cm_test = confusion_matrix_for_display(y_bin_test, test_preds_bin, title=f"{model_name} — held-out test")
    print("\nPooled CV confusion matrix:\n", cm_cv)
    print("\nHeld-out test confusion matrix:\n", cm_test)

    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)
    gbr_explain.fit(X_train_s, y_train)
    run_shap_lime_explanations(model_name, gbr_explain, X_train_s, X_test_s, feature_cols, mode="regression")

    record_standard_model_outputs(
        model_name,
        split_info={
            "train_rows": len(train_idx),
            "test_rows": len(test_idx),
            "train_participants": len(np.unique(groups_train)),
            "test_participants": len(np.unique(groups_test)),
        },
        best_params={"model": model_name, **best_gbr_params},
        train_test_balance=train_test_balance_df,
        cv_folds=cv_results_df,
        cv_summary=cv_summary_df,
        test_summary=test_summary_df,
        wilcoxon={"held_out_merf_gbr_vs_train_mean": w_test},
        confusion_matrices={"cv_pooled": cm_cv, "held_out_test": cm_test},
    )


def run_all(export_excel=True):
    reset_results_store()
    run_merf_gbr_radar()
    run_merf_gbr_androids()
    if export_excel:
        export_results_to_excel()


if __name__ == "__main__":
    run_all()
