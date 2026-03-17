from pathlib import Path
import pandas as pd
import numpy as np


PHQ8_VALUE_COLUMNS = [f"value{i}" for i in range(8)]


def load_csv(file_path: str) -> pd.DataFrame:
    """Load a single CSV file."""
    return pd.read_csv(file_path)


def load_named_csvs(folder_path: str, filenames: list[str]) -> dict[str, pd.DataFrame]:
    """Load multiple named CSVs from a folder."""
    folder = Path(folder_path)
    loaded = {}

    for name in filenames:
        path = folder / name
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        loaded[name] = pd.read_csv(path)

    return loaded


def convert_unix_to_datetime(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert unix timestamp columns to pandas datetime."""
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit="s", errors="coerce")
    return df


def assign_phq8_severity(score):
    """Convert PHQ-8 total score to severity band."""
    if pd.isna(score):
        return np.nan
    if 0 <= score <= 4:
        return "minimal"
    elif 5 <= score <= 9:
        return "mild"
    elif 10 <= score <= 14:
        return "moderate"
    elif 15 <= score <= 19:
        return "moderately_severe"
    elif 20 <= score <= 24:
        return "severe"
    return np.nan


def clean_phq8_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw PHQ-8 questionnaire export."""
    df = df.copy()

    rename_map = {
        "participant_name": "participant_id",
        "name": "questionnaire_name"
    }
    df = df.rename(columns=rename_map)

    unix_cols = [
        "time",
        "time_completed_utc",
        "time_notification",
        "file_modified_on"
    ]
    df = convert_unix_to_datetime(df, unix_cols)

    if "time_completed_local" in df.columns:
        df["completed_at"] = pd.to_datetime(df["time_completed_local"], errors="coerce")
    elif "time_completed_utc" in df.columns:
        df["completed_at"] = df["time_completed_utc"]

    for col in PHQ8_VALUE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["phq8_score"] = df[PHQ8_VALUE_COLUMNS].sum(axis=1, min_count=8)
    df["severity"] = df["phq8_score"].apply(assign_phq8_severity)

    if "project_id" in df.columns:
        df["site"] = df["project_id"].astype(str)

    return df


def select_model_ready_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only columns useful for analysis and later merging."""
    desired_columns = [
        "participant_id",
        "project_id",
        "site",
        "questionnaire_name",
        "version",
        "completed_at",
        "time_completed_utc",
        "phq8_score",
        "severity",
        "file_name",
        "source_id",
        "created_at",
        "updated_at"
    ]
    existing = [col for col in desired_columns if col in df.columns]
    return df[existing].copy()


def aggregate_participant_labels(df: pd.DataFrame, strategy: str = "latest") -> pd.DataFrame:
    """Aggregate multiple PHQ-8 rows per participant."""
    df = df.copy()

    if "completed_at" not in df.columns:
        raise ValueError("completed_at column is required")

    df = df.sort_values(["participant_id", "completed_at"])

    if strategy == "latest":
        return df.groupby("participant_id", as_index=False).tail(1).reset_index(drop=True)

    if strategy == "mean":
        return (
            df.groupby("participant_id", as_index=False)["phq8_score"]
            .mean()
            .rename(columns={"phq8_score": "phq8_score_mean"})
        )

    if strategy == "max":
        return (
            df.groupby("participant_id", as_index=False)["phq8_score"]
            .max()
            .rename(columns={"phq8_score": "phq8_score_max"})
        )

    raise ValueError("strategy must be 'latest', 'mean', or 'max'")