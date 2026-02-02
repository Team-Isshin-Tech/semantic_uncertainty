"""Post-hoc analysis of semantic entropy noise robustness.

Loads se_after_noise.jsonl, validates fields, saves cleaned tables (CSV/JSON),
and produces two scatter plots:
- Plot 1: delta_se vs se_before, color by correctness (green/red)
- Plot 2: delta_se vs mean_noise, color by sigma (scatter, no aggregation)

Usage:
    python analysis/analyze_noise_robustness.py --input path/to/se_after_noise.jsonl \
        --output-dir analysis_outputs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # Headless friendly
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REQUIRED_FIELDS = {
    "question_id",
    "question",
    "generated_answers",
    "mu",
    "sigma",
    "se_before",
    "se_after_samples",
    "se_mean",
    "se_std",
    "delta_se",
    "correctness",
    "mean_noise",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze semantic entropy noise robustness")
    parser.add_argument("--input", required=True, type=Path, help="Path to se_after_noise.jsonl")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis_outputs"),
        help="Directory to write tables and plots",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    missing_rows = 0
    bad_rows = 0

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                bad_rows += 1
                print(f"Skipping line {line_no}: JSON decode error ({exc})")
                continue

            missing = REQUIRED_FIELDS - obj.keys()
            if missing:
                missing_rows += 1
                print(f"Skipping line {line_no}: missing fields {sorted(missing)}")
                continue

            # Basic structural validation
            if not isinstance(obj.get("se_after_samples"), list):
                bad_rows += 1
                print(f"Skipping line {line_no}: se_after_samples is not a list")
                continue
            if len(obj["se_after_samples"]) == 0:
                bad_rows += 1
                print(f"Skipping line {line_no}: se_after_samples is empty")
                continue

            records.append(obj)

    print(f"Loaded records: {len(records)} | missing rows: {missing_rows} | bad rows: {bad_rows}")
    if not records:
        raise ValueError("No valid records loaded from input file")
    return records


def to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)

    # Enforce dtypes
    numeric_cols = [
        "mu",
        "sigma",
        "se_before",
        "se_mean",
        "se_std",
        "delta_se",
        "mean_noise",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["correctness"] = pd.to_numeric(df["correctness"], errors="coerce").astype("Int64")

    # Validate no missing required numeric values
    if df[numeric_cols + ["correctness"]].isnull().any().any():
        null_counts = df[numeric_cols + ["correctness"]].isnull().sum()
        raise ValueError(f"Null or non-numeric values found: {null_counts[null_counts > 0].to_dict()}")

    # Keep correctness categorical but allow bool view for plotting
    df["correctness_bool"] = df["correctness"].astype(int).astype(bool)

    # Ensure se_after_samples remains list and length check (informational)
    df["se_after_len"] = df["se_after_samples"].apply(len)
    print(f"se_after_samples length stats: min={df['se_after_len'].min()} max={df['se_after_len'].max()}")

    return df


def save_tables(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "noise_robustness_table.csv"
    json_path = out_dir / "noise_robustness_table.json"

    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)

    print(f"Saved table: {csv_path}")
    print(f"Saved table: {json_path}")


def plot_delta_vs_se_before(df: pd.DataFrame, out_dir: Path) -> None:
    colors = df["correctness"].map({1: "green", 0: "red"}).fillna("gray")

    plt.figure(figsize=(7, 5))
    plt.scatter(df["se_before"], df["delta_se"], c=colors, alpha=0.7, edgecolors="none")
    plt.xlabel("SE before")
    plt.ylabel("Delta SE (mean - before)")
    plt.title("Delta SE vs SE_before (colored by correctness)")

    # Custom legend
    for label, color in [("Correct", "green"), ("Incorrect", "red")]:
        plt.scatter([], [], c=color, alpha=0.7, label=label)
    plt.legend(title="Correctness")

    out_path = out_dir / "plot_delta_se_vs_se_before.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot: {out_path}")


def plot_delta_vs_mean_noise(df: pd.DataFrame, out_dir: Path) -> None:
    unique_sigmas = sorted(df["sigma"].unique())
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(7, 5))
    for idx, sigma in enumerate(unique_sigmas):
        subset = df[df["sigma"] == sigma]
        plt.scatter(
            subset["mean_noise"],
            subset["delta_se"],
            label=f"sigma={sigma}",
            color=cmap(idx % 10),
            alpha=0.7,
            edgecolors="none",
        )

    plt.xlabel("Mean noise (empirical)")
    plt.ylabel("Delta SE (mean - before)")
    plt.title("Delta SE vs Mean Noise (colored by sigma)")
    plt.legend(title="Sigma")

    out_path = out_dir / "plot_delta_se_vs_mean_noise.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved plot: {out_path}")


def main() -> None:
    args = parse_args()

    records = load_jsonl(args.input)
    df = to_dataframe(records)

    save_tables(df, args.output_dir)
    plot_delta_vs_se_before(df, args.output_dir)
    plot_delta_vs_mean_noise(df, args.output_dir)


if __name__ == "__main__":
    main()
