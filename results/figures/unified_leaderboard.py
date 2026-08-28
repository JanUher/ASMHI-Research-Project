"""Build unified leaderboard and poster comparison figures from results/logs/*.xlsx."""

from __future__ import annotations

from pathlib import Path
import re
import shutil

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

sns.set_theme(style="whitegrid", context="talk")

BASE = Path(__file__).resolve().parents[2]
LOGS = BASE / "results" / "logs"
FIG_DIR = BASE / "results" / "figures" / "Comparison of all models"
POSTER_DIR = BASE / "results" / "figures" / "poster"
TABLE_DIR = BASE / "results" / "tables"
REPR_LEARN_DIR = BASE / "results" / "metrics" / "repr_learn"
KINTSUGI_DIR = BASE / "results" / "metrics" / "kintsugi_health"

TRANSFORMER_DISPLAY_NAMES = {
    "wav2vec2": "Wav2Vec2 (Androids)",
    "hubert": "HuBERT (Androids)",
    "whisper": "Whisper (Androids)",
    "kintsugi_whisper": "Kintsugi Health (HC/PT)",
}

SPEECH_ENCODER_ORDER = ("wav2vec2", "hubert", "whisper", "kintsugi_whisper")
SPEECH_BACKBONE_LABELS = {
    "wav2vec2": "Wav2Vec2",
    "hubert": "HuBERT",
    "whisper": "Whisper",
    "kintsugi_whisper": "Kintsugi Health",
}

KINTSUGI_RADAR_DIRS = (
    KINTSUGI_DIR / "RADAR",
    BASE / "data" / "processed" / "Nick" / "Nick_kinstugi_Health" / "RADAR",
    BASE / "data" / "processed" / "Nick_kinstugi_Health" / "RADAR",
)

WORKBOOKS: list[tuple[str, Path]] = [
    ("gradient_boosting", LOGS / "GB Results" / "gradient_boosting_results.xlsx"),
    ("merf_gbr", LOGS / "GB Results" / "merf_gbr_results.xlsx"),
    ("random_forest", LOGS / "RF Results" / "random_forest_results.xlsx"),
    ("merf_rf", LOGS / "merf_rf_results.xlsx"),
    ("svm", LOGS / "SVM Results" / "svm_results.xlsx"),
    ("merf_svr", LOGS / "SVM Results" / "merf_svr_results.xlsx"),
]

# Human-readable names aligned with results/figures/figures.ipynb MODEL_FILES keys.
DISPLAY_NAME = {
    "RADAR RFC": "Random Forest Classifier",
    "RADAR RFR": "Random Forest Regressor",
    "RADAR GBC": "Gradient Boosting Classifier",
    "RADAR GBR": "Gradient Boosting Regressor",
    "RADAR GPBoost": "GPBoost",
    "RADAR SVC": "Support Vector Classifier",
    "RADAR SVR": "Support Vector Regressor",
    "RADAR MERF-RF": "ME RFRegressor",
    "RADAR MERF-GBR": "ME GBRegressor",
    "RADAR MERF-SVR": "ME SVRegressor",
    "Androids RFC": "Random Forest Classifier",
    "Androids RFR": "Random Forest Regressor",
    "Androids GBC": "Gradient Boosting Classifier",
    "Androids GBR": "Gradient Boosting Regressor",
    "Androids GPBoost": "GPBoost",
    "Androids SVC": "Support Vector Classifier",
    "Androids SVR": "Support Vector Regressor",
    "Androids MERF-RF": "ME RFRegressor",
    "Androids MERF-GBR": "ME GBRegressor",
    "Androids MERF-SVR": "ME SVRegressor",
}

POSTER_CLASSIFIER_ORDER = [
    "Random Forest",
    "SVM",
    "Gradient Boosting Classifier",
    "GPBoost",
]

METRIC_DISPLAY_NAME = {
    "accuracy_mean": "Accuracy",
    "f1_mean": "F1",
    "roc_auc_mean": "ROC-AUC",
    "mae_mean": "MAE",
    "rmse_mean": "RMSE",
    "r2_mean": "R2",
}

BASIC_MERF_PAIRS = [
    ("RADAR GBR", "RADAR MERF-GBR"),
    ("RADAR RFR", "RADAR MERF-RF"),
    ("RADAR SVR", "RADAR MERF-SVR"),
    ("Androids GBC", "Androids MERF-GBR"),
    ("Androids RFC", "Androids MERF-RF"),
    ("Androids SVR", "Androids MERF-SVR"),
]

CLASSIFIER_MODELS = [
    "RADAR GBC",
    "RADAR GPBoost",
    "RADAR RFC",
    "RADAR SVC",
    "Androids GBC",
    "Androids GPBoost",
    "Androids RFC",
    "Androids SVC",
]

POSTER_CATEGORIES = ("classifier", "regressor", "gpboost")

GPBOOST_TEST_CM = {
    "RADAR GPBoost": BASE / "results" / "metrics" / "GradBoost" / "RADAR" / "radar_gpboost_test_confusion_matrix.csv",
    "Androids GPBoost": BASE / "results" / "metrics" / "GradBoost" / "Androids" / "androids_gpboost_test_confusion_matrix.csv",
}


def _model_category(model: str) -> str:
    if "MERF" in model:
        return "merf"
    if model in CLASSIFIER_MODELS:
        return "classifier"
    return "regressor"


def save_fig(fig: plt.Figure, path: Path, dpi: int = 300) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


HEATMAP_TITLE_PAD = 18
HEATMAP_TIGHT_RECT = (0, 0, 1, 0.90)


def finalize_heatmap_figure(
    fig: plt.Figure,
    ax: plt.Axes,
    title: str,
    *,
    ylabel: str | None = "model",
) -> None:
    """Add title spacing above the heatmap so labels are not cramped."""
    ax.set_title(title, pad=HEATMAP_TITLE_PAD)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    fig.tight_layout(rect=HEATMAP_TIGHT_RECT)


def _dataset_from_model(model: str) -> str:
    if model.startswith("RADAR"):
        return "RADAR"
    if model.startswith("Androids"):
        return "Androids"
    return "Other"


def poster_display_name(model: str) -> str:
    """Map log workbook model id to figures.ipynb-style label."""
    return DISPLAY_NAME.get(model, model)


def poster_metric_name(metric: str) -> str:
    """Map cv_summary column (e.g. accuracy_mean) to plot label (e.g. Accuracy)."""
    return METRIC_DISPLAY_NAME.get(
        metric,
        metric.removesuffix("_mean").replace("_", " ").title(),
    )


def rename_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=poster_metric_name)


def load_sheet(path: Path, sheet: str) -> pd.DataFrame:
    if not path.exists():
        print(f"Warning: missing workbook {path}")
        return pd.DataFrame()
    xl = pd.ExcelFile(path)
    if sheet not in xl.sheet_names:
        return pd.DataFrame()
    return pd.read_excel(path, sheet_name=sheet)


def load_unified_tables() -> dict[str, pd.DataFrame]:
    cv_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    wilcoxon_parts: list[pd.DataFrame] = []
    cm_parts: list[pd.DataFrame] = []

    for family, path in WORKBOOKS:
        cv = load_sheet(path, "cv_summary")
        if not cv.empty:
            cv = cv.copy()
            cv["workbook_family"] = family
            cv_parts.append(cv)
        test = load_sheet(path, "test_summary")
        if not test.empty:
            test = test.copy()
            test["workbook_family"] = family
            test_parts.append(test)
        wil = load_sheet(path, "wilcoxon")
        if not wil.empty:
            wil = wil.copy()
            wil["workbook_family"] = family
            wilcoxon_parts.append(wil)
        cm = load_sheet(path, "confusion_matrices")
        if not cm.empty:
            cm = cm.copy()
            cm["workbook_family"] = family
            cm_parts.append(cm)

    cv_summary = (
        pd.concat(cv_parts, ignore_index=True).drop_duplicates(subset=["model"], keep="last")
        if cv_parts
        else pd.DataFrame()
    )
    test_summary = (
        pd.concat(test_parts, ignore_index=True).drop_duplicates(subset=["model"], keep="last")
        if test_parts
        else pd.DataFrame()
    )
    wilcoxon = (
        pd.concat(wilcoxon_parts, ignore_index=True).drop_duplicates(
            subset=["model", "test_name"], keep="last"
        )
        if wilcoxon_parts
        else pd.DataFrame()
    )
    confusion_matrices = (
        pd.concat(cm_parts, ignore_index=True)
        if cm_parts
        else pd.DataFrame()
    )

    if not cv_summary.empty:
        cv_summary["dataset"] = cv_summary["model"].map(_dataset_from_model)
        cv_summary["display_name"] = cv_summary["model"].map(DISPLAY_NAME).fillna(cv_summary["model"])

    return {
        "cv_summary": cv_summary,
        "test_summary": test_summary,
        "wilcoxon": wilcoxon,
        "confusion_matrices": confusion_matrices,
    }


def save_table_csv_md(df: pd.DataFrame, stem: str) -> tuple[Path, Path]:
    """Write a results table as CSV + markdown."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / f"{stem}.csv"
    md_path = TABLE_DIR / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(df.to_markdown(index=False), encoding="utf-8")
    print(f"Saved: {csv_path}\nSaved: {md_path}")
    return csv_path, md_path


def build_leaderboard(cv_summary: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        c
        for c in [
            "accuracy_mean",
            "accuracy_std",
            "f1_mean",
            "f1_std",
            "roc_auc_mean",
            "roc_auc_std",
            "mae_mean",
            "mae_std",
            "rmse_mean",
            "rmse_std",
            "r2_mean",
            "r2_std",
        ]
        if c in cv_summary.columns
    ]
    cols = ["dataset", "model", "display_name", "workbook_family"] + metric_cols
    cols = [c for c in cols if c in cv_summary.columns]
    board = cv_summary[cols].sort_values(["dataset", "roc_auc_mean"], ascending=[True, False])
    board = board.reset_index(drop=True)
    board.insert(3, "model_category", board["model"].map(_model_category))
    return board


def write_leaderboard_tables(leaderboard: pd.DataFrame) -> None:
    """Unified leaderboard + classifier / regressor splits (CSV + markdown)."""
    save_table_csv_md(leaderboard, "unified_leaderboard")

    cls = leaderboard[leaderboard["model_category"].eq("classifier")].copy()
    if not cls.empty:
        save_table_csv_md(cls, "unified_leaderboard_classifiers")

    reg = leaderboard[leaderboard["model_category"].eq("regressor")].copy()
    if not reg.empty:
        save_table_csv_md(reg, "unified_leaderboard_regressors")

    merf = leaderboard[leaderboard["model_category"].eq("merf")].copy()
    if not merf.empty:
        save_table_csv_md(merf, "unified_leaderboard_merf")


def write_wilcoxon_table(wilcoxon: pd.DataFrame) -> None:
    if wilcoxon.empty:
        return
    df = wilcoxon.copy()
    df["dataset"] = df["model"].map(_dataset_from_model)
    df = df.sort_values(["dataset", "model", "p_value"]).reset_index(drop=True)
    save_table_csv_md(df, "wilcoxon_held_out_tests")


def wilcoxon_map(wilcoxon: pd.DataFrame) -> dict[tuple[str, str], dict]:
    if wilcoxon.empty:
        return {}
    out: dict[tuple[str, str], dict] = {}
    for model, grp in wilcoxon.groupby("model"):
        row = grp.sort_values("p_value").iloc[0]
        p = float(row["p_value"])
        out[( _dataset_from_model(model), model)] = {
            "p_value": p,
            "sig_label": "*" if p < 0.05 else "ns",
            "test_name": row.get("test_name", ""),
        }
    return out


def plot_classifier_comparison(cv_summary: pd.DataFrame, wilcoxon: pd.DataFrame) -> Path:
    metrics = ["accuracy_mean", "f1_mean", "roc_auc_mean"]
    plot_df = cv_summary[cv_summary["model"].isin(CLASSIFIER_MODELS)].copy()
    wmap = wilcoxon_map(wilcoxon)

    fig, axes = plt.subplots(1, 3, figsize=(24, 6), sharey=False)
    legend_handles = None
    legend_labels = None
    for ax, metric in zip(axes, metrics):
        sub = plot_df[["dataset", "model", metric]].dropna().copy()
        sns.barplot(data=sub, x="model", y=metric, hue="dataset", ax=ax, legend=(ax is axes[0]))
        if ax is axes[0]:
            legend_handles, legend_labels = ax.get_legend_handles_labels()
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
        ax.set_title(metric.replace("_mean", "").replace("_", " ").title())
        ax.tick_params(axis="x", rotation=35)
        tick_models = [t.get_text() for t in ax.get_xticklabels()]
        for i, model in enumerate(tick_models):
            dataset = "RADAR" if model.startswith("RADAR") else "Androids"
            ann = wmap.get((dataset, model), {}).get("sig_label")
            if not ann:
                continue
            heights = [
                p.get_height()
                for p in ax.patches
                if not np.isnan(p.get_height()) and abs(p.get_x() + p.get_width() / 2 - i) < 0.01
            ]
            if heights:
                ax.text(i, max(heights) + 0.02, ann, ha="center", va="bottom", fontsize=10)
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            title="dataset",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=True,
        )
    fig.suptitle("Classifier CV performance (mean across folds)", y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_fig(fig, FIG_DIR / "all_classifiers_metrics_with_wilcoxon.png")


def plot_basic_vs_mixed_delta(cv_summary: pd.DataFrame) -> Path | None:
    rows = []
    unavailable = []
    lookup = cv_summary.set_index("model")

    pair_specs = [
        ("RADAR", "RADAR GBR", "RADAR MERF-GBR"),
        ("RADAR", "RADAR RFR", "RADAR MERF-RF"),
        ("RADAR", "RADAR SVR", "RADAR MERF-SVR"),
        ("Androids", "Androids GBC", "Androids MERF-GBR"),
        ("Androids", "Androids RFC", "Androids MERF-RF"),
        ("Androids", "Androids SVR", "Androids MERF-SVR"),
    ]
    for dataset, basic, mixed in pair_specs:
        if basic == mixed:
            continue
        if basic not in lookup.index or mixed not in lookup.index:
            unavailable.append((dataset, f"{basic} vs {mixed}", "missing model row"))
            continue
        for metric in ["roc_auc_mean", "mae_mean", "rmse_mean"]:
            b_val = lookup.at[basic, metric] if metric in lookup.columns else np.nan
            m_val = lookup.at[mixed, metric] if metric in lookup.columns else np.nan
            if pd.isna(b_val) or pd.isna(m_val):
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "pair": f"{basic} → {mixed}",
                    "metric": metric.replace("_mean", ""),
                    "delta": float(m_val - b_val),
                }
            )

    delta_long = pd.DataFrame(rows)
    if delta_long.empty:
        print("No computable MERF deltas.")
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=delta_long, x="pair", y="delta", hue="metric", ax=ax)
    ax.axhline(0, color="gray", linewidth=1)
    ax.set_title("Mixed-effects vs baseline (MERF − baseline, CV mean)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return save_fig(fig, FIG_DIR / "basic_vs_mixed_delta_rf_svr_gb.png")


def plot_wilcoxon_summary(wilcoxon: pd.DataFrame) -> list[Path]:
    if wilcoxon.empty:
        return []
    plot_df = wilcoxon.copy()
    plot_df["dataset"] = plot_df["model"].map(_dataset_from_model)
    plot_df = (
        plot_df.sort_values("p_value")
        .groupby(["dataset", "model"], as_index=False)
        .first()
    )
    plot_df["neg_log10_p"] = -np.log10(plot_df["p_value"].clip(lower=1e-300))
    plot_df["poster_model"] = plot_df["model"].map(poster_display_name)

    paths: list[Path] = []
    for dataset in ["RADAR", "Androids"]:
        sub = plot_df[plot_df["dataset"] == dataset].sort_values("neg_log10_p", ascending=True)
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(sub))))
        sns.barplot(
            data=sub,
            y="poster_model",
            x="neg_log10_p",
            color="#4C72B0" if dataset == "RADAR" else "#DD8452",
            ax=ax,
        )
        ax.axvline(-np.log10(0.05), color="red", linestyle="--", linewidth=1, label="p = 0.05")
        ax.set_xlabel("-log10(p)")
        ax.set_title(f"Wilcoxon tests — {dataset} (best p per model)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        slug = dataset.lower()
        paths.append(save_fig(fig, FIG_DIR / f"wilcoxon_pvalues_summary_{slug}.png"))

    legacy = FIG_DIR / "wilcoxon_pvalues_summary.png"
    if legacy.exists():
        legacy.unlink()
    return paths


def _heatmap_color_matrix(
    pivot: pd.DataFrame,
    *,
    higher_better: dict[str, bool] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> pd.DataFrame:
    """Return 0–1 color matrix; annotations use raw pivot values."""
    higher_better = higher_better or {}
    colors = pivot.copy().astype(float)
    for col in pivot.columns:
        vals = pivot[col]
        if vmin is not None and vmax is not None and col in {"accuracy_mean", "f1_mean", "roc_auc_mean"}:
            lo, hi = vmin, vmax
        else:
            valid = vals.dropna()
            if valid.empty:
                colors[col] = np.nan
                continue
            lo, hi = float(valid.min()), float(valid.max())
        if hi == lo:
            colors[col] = 0.5
            continue
        scaled = (vals - lo) / (hi - lo)
        if not higher_better.get(col, True):
            scaled = 1 - scaled
        colors[col] = scaled
    return colors


def plot_leaderboard_heatmap(leaderboard: pd.DataFrame) -> list[Path]:
    paths: list[Path] = []

    cls_metrics = [c for c in ["accuracy_mean", "f1_mean", "roc_auc_mean"] if c in leaderboard.columns]
    if cls_metrics:
        for dataset in ["RADAR", "Androids"]:
            sub = leaderboard[leaderboard["dataset"] == dataset].copy()
            if sub.empty:
                continue
            sub["poster_model"] = sub["display_name"].fillna(sub["model"].map(poster_display_name))
            cls_long = sub.melt(
                id_vars=["poster_model", "dataset"],
                value_vars=cls_metrics,
                var_name="metric",
                value_name="value",
            )
            cls_pivot = cls_long.pivot_table(
                index="poster_model", columns="metric", values="value", aggfunc="first"
            )
            cls_pivot = cls_pivot.dropna(how="all")
            if cls_pivot.empty:
                continue
            cls_pivot = rename_metric_columns(cls_pivot)
            fig_h = max(3.5, 0.55 * len(cls_pivot))
            fig, ax = plt.subplots(figsize=(8, fig_h))
            sns.heatmap(
                cls_pivot,
                annot=True,
                fmt=".3f",
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                ax=ax,
                cbar_kws={"label": "score"},
            )
            finalize_heatmap_figure(
                fig, ax, f"Classification metrics — {dataset} (CV means)"
            )
            slug = dataset.lower()
            paths.append(
                save_fig(fig, FIG_DIR / f"unified_leaderboard_heatmap_classification_{slug}.png")
            )

        legacy_cls = FIG_DIR / "unified_leaderboard_heatmap_classification.png"
        if legacy_cls.exists():
            legacy_cls.unlink()

    reg_metrics = [c for c in ["mae_mean", "rmse_mean", "r2_mean"] if c in leaderboard.columns]
    if reg_metrics:
        higher_better = {
            "mae_mean": False,
            "rmse_mean": False,
            "r2_mean": True,
        }
        for dataset in ["RADAR", "Androids"]:
            reg_board = leaderboard[
                (leaderboard["dataset"] == dataset) & leaderboard[reg_metrics].notna().any(axis=1)
            ].copy()
            if reg_board.empty:
                continue
            reg_board["poster_model"] = reg_board["display_name"].fillna(
                reg_board["model"].map(poster_display_name)
            )
            reg_long = reg_board.melt(
                id_vars=["poster_model", "dataset"],
                value_vars=reg_metrics,
                var_name="metric",
                value_name="value",
            )
            reg_pivot = reg_long.pivot_table(
                index="poster_model", columns="metric", values="value", aggfunc="first"
            )
            reg_pivot = reg_pivot.dropna(how="all")
            if reg_pivot.empty:
                continue
            color_mat = _heatmap_color_matrix(reg_pivot, higher_better=higher_better)
            reg_display = rename_metric_columns(reg_pivot)
            color_display = rename_metric_columns(color_mat)
            fig, ax = plt.subplots(figsize=(8, max(5, 0.4 * len(reg_pivot))))
            sns.heatmap(
                color_display,
                annot=reg_display,
                fmt=".3f",
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                ax=ax,
                cbar_kws={"label": "normalized score (green = better)"},
            )
            finalize_heatmap_figure(
                fig, ax, f"Regression metrics — {dataset} (CV means)"
            )
            slug = dataset.lower()
            paths.append(
                save_fig(fig, FIG_DIR / f"unified_leaderboard_heatmap_regression_{slug}.png")
            )

        legacy_reg = FIG_DIR / "unified_leaderboard_heatmap_regression.png"
        if legacy_reg.exists():
            legacy_reg.unlink()

    legacy = FIG_DIR / "unified_leaderboard_heatmap.png"
    if legacy.exists():
        legacy.unlink()
    return paths


def _speech_encoder_display_name(backbone: str, dataset: str, site: str = "") -> str:
    family = SPEECH_BACKBONE_LABELS.get(backbone, backbone)
    if dataset == "Androids":
        return f"{family} (Androids)"
    if site:
        return f"{family} (RADAR-{site})"
    return f"{family} (RADAR)"


def _parse_repr_summary_stem(stem: str) -> tuple[str, str, str] | None:
    """Return (dataset, site, backbone) from a repr_learn summary filename."""
    if not stem.endswith("_summary") or stem.endswith("_dl_head_no_folds_summary"):
        return None
    if stem.endswith("_embeddings"):
        return None
    base = stem[: -len("_summary")]
    if base.startswith("androids_"):
        backbone = base[len("androids_") :]
        if backbone in SPEECH_BACKBONE_LABELS:
            return "Androids", "", backbone
        return None
    if base.startswith("radar_"):
        rest = base[len("radar_") :]
        for backbone in ("wav2vec2", "hubert", "whisper"):
            token = f"_{backbone}"
            if rest.endswith(token):
                site = rest[: -len(token)]
                if site in {"CIBER_IISPV", "CIBER+IISPV"}:
                    site = "CIBER+IISPV"
                if site:
                    return "RADAR", site, backbone
    return None


def _speech_row_sort_key(row: pd.Series) -> tuple:
    dataset_rank = 0 if row.get("dataset") == "Androids" else 1
    site = str(row.get("site") or "")
    site_rank = {
        "": 0,
        "KCL": 1,
        "CIBER+IISPV": 2,
        "CIBER_IISPV": 2,
        "VUmc": 3,
    }.get(site, 9)
    backbone_rank = {name: i for i, name in enumerate(SPEECH_ENCODER_ORDER)}.get(
        str(row.get("backbone")), 9
    )
    return (dataset_rank, site_rank, backbone_rank)


def load_kintsugi_leaderboard(metrics_dir: Path | None = None) -> pd.DataFrame:
    """Load Androids HC/PT and RADAR DAM-like Whisper held-out test metrics."""
    metrics_dir = Path(metrics_dir or KINTSUGI_DIR)
    rows: list[dict] = []

    hc_pt_path = metrics_dir / "dam_whisper_hc_pt_summary.csv"
    if hc_pt_path.exists():
        df = pd.read_csv(hc_pt_path)
        if not df.empty:
            row = df.iloc[0].to_dict()
            rows.append(
                {
                    "dataset": "Androids",
                    "site": "",
                    "model": _speech_encoder_display_name("kintsugi_whisper", "Androids"),
                    "backbone": "kintsugi_whisper",
                    "approach": "dam_whisper_head",
                    "accuracy_mean": row.get("accuracy"),
                    "f1_mean": row.get("f1"),
                    "roc_auc_mean": row.get("roc_auc"),
                    "eval_protocol": "held_out_test",
                    "source_model": row.get("model", "DAM-like Whisper (HC/PT)"),
                    "n_rows": row.get("n_total"),
                    "n_groups": row.get("n_participants"),
                }
            )
    else:
        print(f"Warning: Kintsugi HC/PT summary missing: {hc_pt_path}")

    radar_dir = next((p for p in KINTSUGI_RADAR_DIRS if p.is_dir()), None)
    if radar_dir is None:
        print("Warning: Kintsugi RADAR metrics directory not found.")
    else:
        for path in sorted(radar_dir.glob("dam_whisper_radar_*_summary.csv")):
            df = pd.read_csv(path)
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            site = str(row.get("sites") or path.stem.replace("dam_whisper_radar_", "").replace("_summary", ""))
            site = site.replace("_", "+") if site == "CIBER_IISPV" else site
            rows.append(
                {
                    "dataset": "RADAR",
                    "site": site,
                    "model": _speech_encoder_display_name("kintsugi_whisper", "RADAR", site),
                    "backbone": "kintsugi_whisper",
                    "approach": "dam_whisper_head",
                    "accuracy_mean": row.get("accuracy"),
                    "f1_mean": row.get("f1"),
                    "roc_auc_mean": row.get("roc_auc"),
                    "eval_protocol": "held_out_test",
                    "source_model": row.get("model", f"DAM-like Whisper (RADAR - {site})"),
                    "n_rows": row.get("n_total"),
                    "n_groups": row.get("n_participants"),
                }
            )

    return pd.DataFrame(rows)


def load_transformer_leaderboard(
    repr_dir: Path | None = None,
    kintsugi_dir: Path | None = None,
) -> pd.DataFrame:
    """Load speech-encoder metrics: linear probes (Androids + RADAR sites) and Kintsugi."""
    repr_dir = Path(repr_dir or REPR_LEARN_DIR)
    rows: list[dict] = []
    if repr_dir.exists():
        for path in sorted(repr_dir.glob("*_summary.csv")):
            parsed = _parse_repr_summary_stem(path.stem)
            if parsed is None:
                continue
            dataset, site, backbone = parsed
            if dataset == "RADAR" and site in {"CIBER", "IISPV"}:
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            row["dataset"] = dataset
            row["site"] = site
            row["backbone"] = backbone
            row["model"] = _speech_encoder_display_name(backbone, dataset, site)
            row["approach"] = "linear_probe"
            row["eval_protocol"] = "5fold_group_cv"
            rows.append(row)
    else:
        print(f"Warning: repr_learn metrics dir missing: {repr_dir}")

    kintsugi = load_kintsugi_leaderboard(kintsugi_dir)
    if not kintsugi.empty:
        rows.extend(kintsugi.to_dict(orient="records"))

    board = pd.DataFrame(rows)
    if board.empty:
        return board
    board["_sort"] = board.apply(_speech_row_sort_key, axis=1)
    board = board.sort_values("_sort").drop(columns="_sort").reset_index(drop=True)
    return board


def plot_transformer_heatmap(
    repr_dir: Path | None = None,
    kintsugi_dir: Path | None = None,
) -> Path | None:
    """Classification heatmap for speech encoders (Androids + RADAR sites + Kintsugi)."""
    board = load_transformer_leaderboard(repr_dir, kintsugi_dir)
    if board.empty:
        print("No speech-encoder metrics found; skipping transformer heatmap.")
        return None

    cls_metrics = [c for c in ["accuracy_mean", "f1_mean", "roc_auc_mean"] if c in board.columns]
    if not cls_metrics:
        print("Speech-encoder summaries lack classification metrics.")
        return None

    long = board.melt(
        id_vars=["model", "dataset", "backbone", "eval_protocol"],
        value_vars=cls_metrics,
        var_name="metric",
        value_name="value",
    )
    pivot = long.pivot_table(index="model", columns="metric", values="value", aggfunc="first")
    pivot = pivot.reindex(board["model"].tolist()).dropna(how="all")
    if pivot.empty:
        return None

    pivot = rename_metric_columns(pivot)

    fig, ax = plt.subplots(figsize=(8, max(3.8, 0.55 * len(pivot))))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": "score"},
    )
    ax.tick_params(axis="x", labelsize=10)
    finalize_heatmap_figure(
        fig,
        ax,
        "Classification metrics — speech encoders",
        ylabel="encoder / cohort",
    )
    out = save_fig(fig, FIG_DIR / "unified_leaderboard_heatmap_speech_encoders.png")

    legacy = FIG_DIR / "unified_leaderboard_heatmap_transformers_androids.png"
    if legacy.exists():
        legacy.unlink()

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    keep_cols = [
        c
        for c in [
            "dataset",
            "site",
            "model",
            "backbone",
            "approach",
            "eval_protocol",
            "n_rows",
            "n_groups",
            "accuracy_mean",
            "accuracy_std",
            "f1_mean",
            "f1_std",
            "roc_auc_mean",
            "roc_auc_std",
            "source_model",
        ]
        if c in board.columns
    ]
    export = board[keep_cols].copy()
    export.to_csv(TABLE_DIR / "transformer_leaderboard.csv", index=False)
    (TABLE_DIR / "transformer_leaderboard.md").write_text(
        export.to_markdown(index=False), encoding="utf-8"
    )
    print(f"Saved: {TABLE_DIR / 'transformer_leaderboard.csv'}")
    print(f"Saved: {TABLE_DIR / 'transformer_leaderboard.md'}")
    return out


def pick_best_models_per_category(leaderboard: pd.DataFrame) -> pd.DataFrame:
    """Best classifier and regressor per dataset, plus GPBoost (mixed-effects boosting)."""
    rows: list[dict] = []
    for dataset in ["RADAR", "Androids"]:
        sub = leaderboard[leaderboard["dataset"] == dataset].copy()
        if sub.empty:
            continue

        cls = sub[sub["model"].isin(CLASSIFIER_MODELS)].dropna(subset=["roc_auc_mean"])
        if not cls.empty:
            best = cls.loc[cls["roc_auc_mean"].idxmax()]
            rows.append(
                {
                    "dataset": dataset,
                    "category": "classifier",
                    "model": best["model"],
                    "display_name": best.get("display_name", poster_display_name(best["model"])),
                    "selection_metric": "roc_auc_mean",
                    "selection_value": float(best["roc_auc_mean"]),
                }
            )

        reg = sub[
            sub["model"].map(_model_category).eq("regressor")
        ].dropna(subset=["mae_mean"])
        if not reg.empty:
            best = reg.loc[reg["mae_mean"].idxmin()]
            rows.append(
                {
                    "dataset": dataset,
                    "category": "regressor",
                    "model": best["model"],
                    "display_name": best.get("display_name", poster_display_name(best["model"])),
                    "selection_metric": "mae_mean",
                    "selection_value": float(best["mae_mean"]),
                }
            )

        gp_name = f"{dataset} GPBoost"
        gp = sub[sub["model"] == gp_name]
        if not gp.empty:
            row = gp.iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "category": "gpboost",
                    "model": row["model"],
                    "display_name": row.get("display_name", poster_display_name(row["model"])),
                    "selection_metric": "roc_auc_mean",
                    "selection_value": float(row["roc_auc_mean"]) if pd.notna(row.get("roc_auc_mean")) else float("nan"),
                }
            )

    return pd.DataFrame(rows)


def _confusion_matrix_from_pred_csv(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    cm = np.zeros((2, 2), dtype=int)
    for i, true_key in enumerate(("true_0", "true_1")):
        if true_key not in df.index:
            continue
        for j, pred_key in enumerate(("pred_0", "pred_1")):
            if pred_key in df.columns:
                cm[i, j] = int(df.loc[true_key, pred_key])
    return cm


def _confusion_matrix_array(
    confusion_matrices: pd.DataFrame,
    model_name: str,
    matrix_name: str = "held_out_test",
) -> np.ndarray | None:
    if model_name in GPBOOST_TEST_CM:
        loaded = _confusion_matrix_from_pred_csv(GPBOOST_TEST_CM[model_name])
        if loaded is not None:
            return loaded
    if confusion_matrices.empty:
        return None
    sub = confusion_matrices[
        (confusion_matrices["model"] == model_name)
        & (confusion_matrices["matrix"] == matrix_name)
    ]
    if sub.empty:
        return None
    label_map = {"control_0": 0, "depressed_1": 1}
    cm = np.zeros((2, 2), dtype=int)
    for _, row in sub.iterrows():
        i = label_map.get(row["true_label"])
        j = label_map.get(row["pred_label"])
        if i is None or j is None:
            continue
        cm[i, j] = int(row["count"])
    return cm


def _plot_confusion_matrix_ax(
    ax: plt.Axes,
    cm: np.ndarray,
    *,
    title: str | None = None,
    display_labels: tuple[str, str] = ("control (0)", "depressed (1)"),
) -> None:
    """Draw a 2×2 confusion matrix filling the axes (original layout)."""
    cm = np.asarray(cm, dtype=int)
    vmax = max(int(cm.max()), 1)
    ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=vmax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.xaxis.set_major_locator(FixedLocator([0, 1]))
    ax.yaxis.set_major_locator(FixedLocator([0, 1]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(list(display_labels))
    ax.set_yticklabels(list(display_labels))
    ax.set_xlabel("Predicted label", labelpad=10)
    ax.set_ylabel("True label", labelpad=10)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    ax.grid(False)
    ax.tick_params(which="both", length=0, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(True)
    if title:
        ax.set_title(title)


def _save_confusion_figure(
    fig: plt.Figure,
    outfile: Path,
    dpi: int = 300,
    *,
    subplots_adjust: dict[str, float] | None = None,
) -> Path:
    """Save confusion matrix without bbox_inches='tight' (it rasterizes a white stripe over the heatmap)."""
    outfile.parent.mkdir(parents=True, exist_ok=True)
    adjust = subplots_adjust or {"left": 0.24, "bottom": 0.18, "right": 0.96, "top": 0.86}
    fig.subplots_adjust(**adjust)
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {outfile}")
    return outfile


def export_confusion_matrix(
    confusion_matrices: pd.DataFrame,
    model_name: str,
    matrix_name: str = "held_out_test",
    *,
    category: str | None = None,
    dataset: str | None = None,
    outfile: Path | None = None,
) -> Path | None:
    cm = _confusion_matrix_array(confusion_matrices, model_name, matrix_name)
    if cm is None:
        print(f"No confusion matrix for {model_name} / {matrix_name}")
        return None

    with plt.rc_context({"axes.grid": False}):
        with sns.axes_style("white"):
            fig, ax = plt.subplots(figsize=(5, 4))
            label = poster_display_name(model_name)
            title = f"{label} — {matrix_name.replace('_', ' ')}"
            if category and dataset:
                title = f"{dataset} | {category.title()}\n{label} (held-out test)"
            _plot_confusion_matrix_ax(ax, cm, title=title)
    if outfile is None:
        slug = model_name.replace(" ", "_").lower()
        if category and dataset:
            outfile = POSTER_DIR / f"confusion_matrix_{dataset.lower()}_{category}_{slug}.png"
        else:
            outfile = POSTER_DIR / f"confusion_matrix_{slug}.png"
    return _save_confusion_figure(fig, outfile)


def export_poster_confusion_grid(
    confusion_matrices: pd.DataFrame,
    best_per_category: pd.DataFrame,
    matrix_name: str = "held_out_test",
) -> Path | None:
    if best_per_category.empty:
        return None

    panels = []
    for _, row in best_per_category.iterrows():
        cm = _confusion_matrix_array(confusion_matrices, row["model"], matrix_name)
        if cm is not None:
            panels.append(row.to_dict() | {"cm": cm})

    if not panels:
        print("No confusion matrices available for best-per-category grid.")
        return None

    n = len(panels)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    with plt.rc_context({"axes.grid": False}):
        with sns.axes_style("white"):
            fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.0 * nrows))
            axes = np.atleast_1d(axes).ravel()

            for ax, panel in zip(axes, panels):
                label = panel.get("display_name", poster_display_name(panel["model"]))
                cat = str(panel["category"])
                cat_label = "GPBoost" if cat.lower() == "gpboost" else cat.title()
                title = (
                    f"{panel['dataset']} — {cat_label}\n"
                    f"{label}"
                )
                _plot_confusion_matrix_ax(ax, panel["cm"], title=title)
                ax.title.set_fontsize(10)
                ax.tick_params(labelsize=9)

            for ax in axes[len(panels) :]:
                ax.axis("off")

            fig.suptitle(
                "Held-out test confusion matrices — best model per category",
                y=0.98,
            )

    return _save_confusion_figure(
        fig,
        POSTER_DIR / "confusion_matrices_best_per_category.png",
        subplots_adjust={
            "left": 0.14,
            "bottom": 0.12,
            "right": 0.98,
            "top": 0.88,
            "wspace": 0.38,
            "hspace": 0.52,
        },
    )


def export_poster_confusion_matrices(
    confusion_matrices: pd.DataFrame,
    leaderboard: pd.DataFrame,
) -> pd.DataFrame:
    """Export individual + grid confusion matrices for best model in each category."""
    best = pick_best_models_per_category(leaderboard)
    if best.empty:
        print("No best-per-category selections.")
        return best

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    save_table_csv_md(best, "poster_best_models_per_category")
    print(best.to_string(index=False))

    for _, row in best.iterrows():
        export_confusion_matrix(
            confusion_matrices,
            row["model"],
            category=row["category"],
            dataset=row["dataset"],
        )

    export_poster_confusion_grid(confusion_matrices, best)
    return best


def export_shap_beeswarm(model_name: str = "Androids RFC") -> Path | None:
    try:
        import shap
    except ImportError:
        print("shap not installed; skipping SHAP export")
        return None

    df = pd.read_csv(BASE / "data" / "processed" / "androids_model_dataset_basic.csv")
    meta_cols = [
        "file_path",
        "file",
        "file_stem",
        "bdi_score",
        "depressed",
        "fold",
        "speech_type",
        "subgroup_from_path",
    ]
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = df.dropna(subset=feature_cols + ["depressed", "file_stem"]).copy()
    X = df[feature_cols].values
    y = df["depressed"].values
    groups = df["file_stem"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]

    model = RandomForestClassifier(
        n_estimators=200,
        max_features="sqrt",
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y[train_idx])

    bg = shap.sample(X_train, min(200, len(X_train)), random_state=42)
    X_shap = X_test[: min(45, len(X_test))]
    explainer = shap.Explainer(model, bg)
    shap_values = explainer(X_shap)

    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, pd.DataFrame(X_shap, columns=feature_cols), show=False)
    plt.title(f"{model_name} — SHAP beeswarm (held-out test)")
    plt.tight_layout()
    return save_fig(fig, POSTER_DIR / f"shap_beeswarm_{model_name.replace(' ', '_').lower()}.png")


def write_table_1() -> tuple[Path, Path]:
    radar = pd.read_csv(BASE / "data" / "processed" / "radar_model_dataset_raw_features.csv")
    androids = pd.read_csv(BASE / "data" / "processed" / "androids_model_dataset_basic.csv")

    radar_acoustic = [
        c
        for c in radar.columns
        if c
        not in {
            "File",
            "participant_id",
            "Dataset",
            "Language",
            "Task",
            "recording_date",
            "Age",
            "Gender",
            "Education_Years",
            "Height",
            "phq8_score",
            "Clip_Duration",
        }
    ]
    androids_acoustic = [
        c
        for c in androids.columns
        if c
        not in {
            "file_path",
            "file",
            "file_stem",
            "bdi_score",
            "depressed",
            "fold",
            "speech_type",
            "subgroup_from_path",
        }
    ]

    rows = [
        {
            "dataset": "RADAR-MDD",
            "unit_of_analysis": "Recording (clip)",
            "group_key": "participant_id",
            "n_units": len(radar),
            "n_groups": radar["participant_id"].nunique(),
            "label_continuous": "phq8_score",
            "label_binary": "depressed (PHQ-8 ≥ 10)",
            "n_depressed": int((radar["phq8_score"] >= 10).sum()),
            "n_control": int((radar["phq8_score"] < 10).sum()),
            "speech_tasks": "Scripted + Unscripted",
            "features": f"{len(radar_acoustic)} acoustic (+ demographics in MERF Z)",
            "modality": "Tabular acoustics",
        },
        {
            "dataset": "Androids",
            "unit_of_analysis": "Recording",
            "group_key": "file_stem",
            "n_units": len(androids),
            "n_groups": androids["file_stem"].nunique(),
            "label_continuous": "bdi_score",
            "label_binary": "depressed (BDI-derived)",
            "n_depressed": int((androids["depressed"] == 1).sum()),
            "n_control": int((androids["depressed"] == 0).sum()),
            "speech_tasks": "read + interview",
            "features": f"{len(androids_acoustic)} acoustic (MFCC, pitch, energy)",
            "modality": "Tabular acoustics (+ WAV for transformers)",
        },
    ]
    table = pd.DataFrame(rows)
    save_table_csv_md(table, "table1_datasets")
    return TABLE_DIR / "table1_datasets.csv", TABLE_DIR / "table1_datasets.md"


def write_table_2() -> tuple[Path, Path]:
    rows = [
        {"step": "Train/test split", "detail": "GroupShuffleSplit — 80% train pool / 20% held-out test"},
        {"step": "Group key", "detail": "RADAR: participant_id; Androids: file_stem"},
        {"step": "Cross-validation", "detail": "5-fold GroupKFold on train pool only"},
        {"step": "Classifier tuning", "detail": "GridSearchCV, scoring = ROC-AUC (GBC, RFC, GPBoost, SVC)"},
        {"step": "Regressor tuning", "detail": "GridSearchCV, scoring = neg-RMSE (GBR, RFR, SVR)"},
        {"step": "SVM models", "detail": "SVC (classification), SVR (regression), MERF-SVR (mixed-effects); RBF kernel, scaled features"},
        {"step": "Binary from regression", "detail": "RADAR: ŷ ≥ 10; Androids: ŷ ≥ BDI train midpoint"},
        {"step": "MERF", "detail": "Fixed-effects model tuned on X; MERF fit per CV fold with Z covariates"},
        {"step": "Wilcoxon baseline", "detail": "Classifiers: stratified dummy; Regressors/MERF: train-mean"},
        {"step": "Fairness", "detail": "AIF360 disparate impact + statistical parity (gender)"},
        {"step": "Held-out reporting", "detail": "test_summary + confusion_matrices sheets in Excel logs"},
    ]
    table = pd.DataFrame(rows)
    save_table_csv_md(table, "table2_evaluation_protocol")
    return TABLE_DIR / "table2_evaluation_protocol.csv", TABLE_DIR / "table2_evaluation_protocol.md"


def _speech_metric(row: pd.Series, name: str) -> float | None:
    for key in (f"{name}_mean", name):
        if key in row and pd.notna(row[key]):
            return float(row[key])
    return None


def write_results_best_snapshot(
    leaderboard: pd.DataFrame,
    speech_board: pd.DataFrame,
) -> pd.DataFrame:
    """Compact winners table: tabular best-per-category plus comparable encoder rows."""
    best = pick_best_models_per_category(leaderboard)
    family_label = {
        "classifier": "classifier",
        "regressor": "regressor",
        "gpboost": "mixed effect model",
        "merf": "mixed effect model",
    }
    rows: list[dict] = []
    for _, row in best.iterrows():
        src = leaderboard[leaderboard["model"] == row["model"]]
        src = src.iloc[0] if not src.empty else row
        rows.append(
            {
                "dataset": row["dataset"],
                "family": family_label.get(str(row["category"]), str(row["category"])),
                "model": src.get("display_name", row.get("display_name", row["model"])),
                "protocol": "grouped 80/20; 5-fold GroupKFold on train",
                "accuracy": src.get("accuracy_mean"),
                "f1": src.get("f1_mean"),
                "roc_auc": src.get("roc_auc_mean"),
                "mae": src.get("mae_mean"),
                "selection": f"{row['selection_metric']}={row['selection_value']:.3f}",
            }
        )

    if not speech_board.empty:
        probes = speech_board[speech_board.get("approach", speech_board.get("eval_protocol", "")) != ""]
        androids_probe = speech_board[
            (speech_board["dataset"] == "Androids")
            & (speech_board["backbone"].isin(["wav2vec2", "hubert", "whisper"]))
        ]
        if not androids_probe.empty:
            auc = androids_probe.apply(lambda r: _speech_metric(r, "roc_auc"), axis=1)
            pick = androids_probe.loc[auc.idxmax()]
            rows.append(
                {
                    "dataset": "Androids",
                    "family": "encoder probe",
                    "model": pick.get("model"),
                    "protocol": "5-fold GroupKFold on all 224 clips",
                    "accuracy": _speech_metric(pick, "accuracy"),
                    "f1": _speech_metric(pick, "f1"),
                    "roc_auc": _speech_metric(pick, "roc_auc"),
                    "mae": np.nan,
                    "selection": "max ROC-AUC among Androids probes",
                }
            )
        kcl_probe = speech_board[
            (speech_board["dataset"] == "RADAR")
            & (speech_board["site"].astype(str).isin(["KCL", "KCL"]))
            & (speech_board["backbone"].isin(["wav2vec2", "hubert", "whisper"]))
        ]
        if kcl_probe.empty:
            kcl_probe = speech_board[
                (speech_board["dataset"] == "RADAR")
                & speech_board["site"].astype(str).str.contains("KCL", case=False, na=False)
                & (speech_board["backbone"].isin(["wav2vec2", "hubert", "whisper"]))
            ]
        if not kcl_probe.empty:
            auc = kcl_probe.apply(lambda r: _speech_metric(r, "roc_auc"), axis=1)
            pick = kcl_probe.loc[auc.idxmax()]
            rows.append(
                {
                    "dataset": "RADAR-KCL",
                    "family": "encoder probe",
                    "model": pick.get("model"),
                    "protocol": "5-fold GroupKFold on Nick KCL embeddings",
                    "accuracy": _speech_metric(pick, "accuracy"),
                    "f1": _speech_metric(pick, "f1"),
                    "roc_auc": _speech_metric(pick, "roc_auc"),
                    "mae": np.nan,
                    "selection": "max ROC-AUC among RADAR-KCL probes",
                }
            )
        for _, pick in speech_board[speech_board["backbone"] == "kintsugi_whisper"].iterrows():
            site = str(pick.get("site") or "")
            if pick.get("dataset") == "RADAR" and site not in {"", "KCL", "KCL"}:
                if "KCL" not in site:
                    continue
            rows.append(
                {
                    "dataset": "Androids" if pick.get("dataset") == "Androids" else f"RADAR-{site or 'KCL'}",
                    "family": "Kintsugi",
                    "model": pick.get("model"),
                    "protocol": "grouped 80/20, no inner CV",
                    "accuracy": _speech_metric(pick, "accuracy"),
                    "f1": _speech_metric(pick, "f1"),
                    "roc_auc": _speech_metric(pick, "roc_auc"),
                    "mae": np.nan,
                    "selection": "held-out test",
                }
            )

    out = pd.DataFrame(rows)
    for col in ("accuracy", "f1", "roc_auc", "mae"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(3)
    save_table_csv_md(out, "results_best_model_snapshot")
    return out


def plot_classification_metric_bars(
    leaderboard: pd.DataFrame,
    speech_board: pd.DataFrame,
) -> list[Path]:
    """Two grouped bar charts (RADAR, Androids) for Accuracy, F1, and ROC-AUC."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def _rows_for_dataset(dataset: str) -> pd.DataFrame:
        recs: list[dict] = []
        tab = leaderboard[leaderboard["dataset"] == dataset]
        tab = tab[~tab["model"].astype(str).str.contains(r"FT[- ]?Transformer", case=False, na=False)]
        for _, r in tab.iterrows():
            if pd.isna(r.get("roc_auc_mean")):
                continue
            recs.append(
                {
                    "model": r.get("display_name", r["model"]),
                    "Accuracy": r.get("accuracy_mean"),
                    "F1": r.get("f1_mean"),
                    "ROC-AUC": r.get("roc_auc_mean"),
                }
            )
        if not speech_board.empty:
            sp = speech_board[speech_board["dataset"] == dataset].copy()
            if dataset == "RADAR":
                sp = sp[sp["site"].astype(str).str.contains("KCL", case=False, na=False)]
            for _, r in sp.iterrows():
                recs.append(
                    {
                        "model": r.get("model"),
                        "Accuracy": _speech_metric(r, "accuracy"),
                        "F1": _speech_metric(r, "f1"),
                        "ROC-AUC": _speech_metric(r, "roc_auc"),
                    }
                )
        df = pd.DataFrame(recs).dropna(subset=["ROC-AUC"])
        if df.empty:
            return df
        df = df.drop_duplicates(subset=["model"], keep="first")
        df = df.sort_values("ROC-AUC", ascending=False)
        return df.melt(id_vars=["model"], value_vars=["Accuracy", "F1", "ROC-AUC"], var_name="metric", value_name="score")

    for dataset in ("RADAR", "Androids"):
        long = _rows_for_dataset(dataset)
        if long.empty:
            continue
        n_models = long["model"].nunique()
        fig, ax = plt.subplots(figsize=(max(10, 0.55 * n_models + 4), 5.8))
        sns.barplot(data=long, x="model", y="score", hue="metric", ax=ax)
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        title = "RADAR-KCL" if dataset == "RADAR" else "Androids"
        ax.set_title(f"{title}: Accuracy, F1, and ROC-AUC")
        ax.legend(title="", loc="upper right")
        plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
        fig.tight_layout()
        path = FIG_DIR / f"metric_bars_{dataset.lower()}.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")
        paths.append(path)
    return paths


def _parse_gender_age_group(value: object) -> tuple[str | None, str | None]:
    s = str(value).strip().lower().replace("-", " ").replace("_", " ")
    age = None
    if "young" in s:
        age = "young"
    elif "middle" in s:
        age = "middle"
    elif "older" in s or "old" in s:
        age = "older"
    gender = None
    if "female" in s or "gender 0" in s or s.endswith("0"):
        if "male" in s and "female" not in s:
            gender = "male"
        elif "female" in s or "gender 0" in s:
            gender = "female"
    if gender is None:
        if "male" in s:
            gender = "male"
        elif "gender 1" in s or s.endswith("1"):
            gender = "male"
    return age, gender


def _model_family_label(model: str) -> str:
    m = str(model)
    if "MERF" in m or m.startswith("ME "):
        if "SVR" in m or "SVM" in m:
            return "MERF-SVR"
        if "GB" in m:
            return "MERF-GBR"
        return "MERF-RF"
    if "GPBoost" in m or "GPBoost" in m:
        return "GPBoost"
    if "RFC" in m or "RFR" in m:
        return "Random Forest"
    if "SVC" in m or "SVR" in m:
        return "SVM"
    if "GBC" in m or "GBR" in m:
        return "Gradient Boosting"
    return "Other"


def load_gender_age_subgroups() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _family, path in WORKBOOKS:
        df = load_sheet(path, "subgroup_gender_age")
        if not df.empty:
            parts.append(df)
    csv_paths = [
        LOGS / "merf_rf" / "subgroup_gender_age.csv",
        LOGS / "merf_svr" / "subgroup_gender_age.csv",
        BASE / "results" / "metrics" / "merf_gbr" / "subgroup_gender_age.csv",
    ]
    for path in csv_paths:
        if path.exists():
            parts.append(pd.read_csv(path))
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    if "group" not in df.columns:
        for alt in ("group", "subgroup", "stratum"):
            if alt in df.columns:
                df = df.rename(columns={alt: "group"})
                break
    if "n" not in df.columns:
        for alt in ("n_samples", "count", "n_rows"):
            if alt in df.columns:
                df = df.rename(columns={alt: "n"})
                break
    df["model"] = df["model"].astype(str)
    parsed = df["group"].map(_parse_gender_age_group)
    df["age"] = parsed.map(lambda x: x[0])
    df["gender"] = parsed.map(lambda x: x[1])
    df = df.dropna(subset=["age", "gender"])
    df["stratum"] = df["age"] + " × " + df["gender"]
    df["dataset"] = df["model"].map(_dataset_from_model)
    df["family"] = df["model"].map(_model_family_label)
    df["is_merf"] = df["family"].str.startswith("MERF")
    df["class"] = np.where(df["is_merf"], "MERF", "Conventional")
    for col in ("n", "accuracy", "f1", "roc_auc"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(subset=["model", "group"], keep="last")
    drop = df["model"].str.contains(r"FT[- ]?Transformer", case=False, na=False)
    return df.loc[~drop].copy()


def plot_intersectional_merf_vs_conventional() -> pd.DataFrame:
    """Average gender×age performance: MERF vs conventional, and by backbone family."""
    df = load_gender_age_subgroups()
    if df.empty:
        print("No subgroup_gender_age sheets found; skipping intersectional comparison.")
        return df

    min_n = df["n"].where(df["dataset"] == "RADAR", df["n"])
    keep = ~((df["dataset"] == "RADAR") & (df["n"] < 20))
    keep &= ~((df["dataset"] == "Androids") & (df["n"] < 4))
    df = df.loc[keep].copy()

    def _wmean(g: pd.DataFrame, col: str) -> float:
        sub = g.dropna(subset=[col, "n"])
        if sub.empty or sub["n"].sum() <= 0:
            return float("nan")
        return float(np.average(sub[col], weights=sub["n"]))

    family_rows = []
    for (dataset, family), g in df.groupby(["dataset", "family"]):
        family_rows.append(
            {
                "dataset": dataset,
                "family": family,
                "class": "MERF" if str(family).startswith("MERF") else "Conventional",
                "accuracy": _wmean(g, "accuracy"),
                "f1": _wmean(g, "f1"),
                "roc_auc": _wmean(g, "roc_auc"),
                "n_cells": len(g),
            }
        )
    family_table = pd.DataFrame(family_rows)
    save_table_csv_md(family_table.round(3), "intersectional_family_means")

    stratum_rows = []
    for (dataset, stratum, cls), g in df.groupby(["dataset", "stratum", "class"]):
        stratum_rows.append(
            {
                "dataset": dataset,
                "stratum": stratum,
                "class": cls,
                "accuracy": _wmean(g, "accuracy"),
                "f1": _wmean(g, "f1"),
                "roc_auc": _wmean(g, "roc_auc"),
                "n": int(g["n"].sum()),
            }
        )
    stratum_table = pd.DataFrame(stratum_rows)
    save_table_csv_md(stratum_table.round(3), "intersectional_merf_vs_conventional")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not stratum_table.empty:
        g = sns.catplot(
            data=stratum_table,
            x="stratum",
            y="roc_auc",
            hue="class",
            col="dataset",
            kind="bar",
            height=4.8,
            aspect=1.15,
            sharey=True,
        )
        g.set(ylim=(0.4, 1.0))
        g.set_axis_labels("", "ROC-AUC")
        g.set_titles("{col_name}")
        for ax in g.axes.flat:
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        g.fig.suptitle("Gender × age ROC-AUC: MERF vs conventional models", y=1.04)
        path = FIG_DIR / "intersectional_merf_vs_conventional_roc_auc.png"
        g.fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(g.fig)
        print(f"Saved: {path}")

    if not family_table.empty:
        long = family_table.melt(
            id_vars=["dataset", "family"],
            value_vars=["accuracy", "f1", "roc_auc"],
            var_name="metric",
            value_name="score",
        )
        long["metric"] = long["metric"].map(
            {"accuracy": "Accuracy", "f1": "F1", "roc_auc": "ROC-AUC"}
        )
        g = sns.catplot(
            data=long,
            x="family",
            y="score",
            hue="metric",
            col="dataset",
            kind="bar",
            height=4.8,
            aspect=1.2,
        )
        g.set(ylim=(0, 1))
        g.set_axis_labels("", "Score")
        g.set_titles("{col_name}")
        for ax in g.axes.flat:
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        g.fig.suptitle("Intersectional (gender × age) mean scores by model family", y=1.04)
        path = FIG_DIR / "intersectional_scores_by_family.png"
        g.fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(g.fig)
        print(f"Saved: {path}")
    return stratum_table


def copy_confusion_matrices_for_results() -> None:
    src = POSTER_DIR / "confusion_matrices_best_per_category.png"
    if src.exists():
        dest = FIG_DIR / "confusion_matrices_best_per_category.png"
        shutil.copy2(src, dest)
        print(f"Copied confusion-matrix grid to {dest}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    POSTER_DIR.mkdir(parents=True, exist_ok=True)

    tables = load_unified_tables()
    cv_summary = tables["cv_summary"]
    if cv_summary.empty:
        raise RuntimeError("No cv_summary rows found in results/logs workbooks.")

    leaderboard = build_leaderboard(cv_summary)
    write_leaderboard_tables(leaderboard)
    write_wilcoxon_table(tables["wilcoxon"])
    print(leaderboard[["dataset", "model", "model_category", "roc_auc_mean", "f1_mean", "mae_mean"]].to_string())

    plot_classifier_comparison(cv_summary, tables["wilcoxon"])
    plot_basic_vs_mixed_delta(cv_summary)
    plot_wilcoxon_summary(tables["wilcoxon"])
    plot_leaderboard_heatmap(leaderboard)
    plot_transformer_heatmap()

    print("\n=== Poster: best model per category (confusion matrices) ===")
    best_by_category = export_poster_confusion_matrices(tables["confusion_matrices"], leaderboard)

    if not best_by_category.empty:
        shap_row = best_by_category[
            best_by_category["category"].eq("classifier")
        ].sort_values("selection_value", ascending=False)
        if not shap_row.empty:
            export_shap_beeswarm(model_name=str(shap_row.iloc[0]["model"]))
    else:
        best_model = leaderboard.sort_values("roc_auc_mean", ascending=False).iloc[0]["model"]
        print(f"\nFallback best model by CV ROC-AUC: {best_model}")
        export_confusion_matrix(tables["confusion_matrices"], str(best_model))
        export_shap_beeswarm(model_name=str(best_model))

    write_table_1()
    write_table_2()

    speech_board = load_transformer_leaderboard()
    write_results_best_snapshot(leaderboard, speech_board)
    plot_classification_metric_bars(leaderboard, speech_board)
    plot_intersectional_merf_vs_conventional()
    copy_confusion_matrices_for_results()
    print("\nDone.")


if __name__ == "__main__":
    main()
