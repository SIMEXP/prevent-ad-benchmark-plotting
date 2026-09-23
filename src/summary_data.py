"""Load and rank the downstream summary table (summary_classification.tsv).

The table is written by downstream_stats.make_summary_table. Each row is one
(foundation model, experiment, feature, target, ...) combination with, per
metric, its mean over splits (`ACCURACY`), a 95% CI (`ACCURACY_CI_LOW/HIGH`) and
tests against the reference classifiers (`ACCURACY_SIG_VS_FC`, ...).

Experiments:
  - "adaptation": frozen foundation model
  - "transfer":   fine-tuned foundation model
  - "baseline":   non-foundation-model features (functional connectivity,
                  timeseries)
  - "chance":     dummy classifier
"""

from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = [("accuracy", "Accuracy"), ("auc", "AUC"), ("precision", "Precision")]
# Superset of METRICS: F1 is only shown in the baseline comparison figure.
ALL_METRICS = METRICS + [("f1", "F1")]

REFERENCE_EXPERIMENTS = {"baseline", "chance"}
FC = "Functional connectivity"
DUMMY = "dummy"
T1 = "T1 (mean-pooling)"

# Bars are ranked by accuracy unless the target is listed here.
DEFAULT_SORT_METRIC = "accuracy"
TARGET_SORT_METRIC = {
    "MCI status": "precision",
    "β-amyloid SUVR > 1.26": "precision",
}

# (foundation model, voxelwise normalization, timeseries extraction) whose
# learning curves converge cleanly; other variations are dropped.
SOUND_VARIATIONS = {
    ("brainharmonix", "yes", "gigaconnectome"),
    ("brainlm650M", "no", "gigaconnectome"),
}
NORMALIZATION_LABELS = {"yes": "global norm", "no": "no global norm"}


def load_summary(path: Path) -> pd.DataFrame:
    """Read summary_classification.tsv, keeping the literal "N/A" strings."""
    return pd.read_csv(path, sep="\t", keep_default_na=False, na_values=[""])


def prepare_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Drop SVM rows and variations with unreliable learning curves, and expose
    each metric as lowercase `metric`, `metric_low` and `metric_high` columns."""
    df = df.copy()
    for key, _ in ALL_METRICS:
        upper = key.upper()
        df[key] = df[upper]
        df[f"{key}_low"] = df[f"{upper}_CI_LOW"]
        df[f"{key}_high"] = df[f"{upper}_CI_HIGH"]

    # T1 rows carry normalization "N/A" (it doesn't apply to T1), so they are
    # matched on model + extraction only.
    sound_without_norm = {(model, extraction) for model, _, extraction in SOUND_VARIATIONS}
    keep_model = [
        (model, norm, extraction) in SOUND_VARIATIONS
        or (norm == "N/A" and (model, extraction) in sound_without_norm)
        for model, norm, extraction in zip(
            df["Foundation Model"],
            df["Voxelwise Normalization"],
            df["Timeseries Extraction"],
        )
    ]
    is_reference = df["Experiment"].isin(REFERENCE_EXPERIMENTS)
    return df[
        (df["Classifier"] != "SVM")
        & (is_reference | pd.Series(keep_model, index=df.index))
    ]


def significance_marks(bars, metric, fc, dummy):
    """"*" for each bar significantly above the functional connectivity baseline.

    No marks at all if that baseline doesn't itself beat chance, since clearing
    it would then not be a meaningful bar.
    """
    if fc[metric] < dummy[metric]:
        return [""] * len(bars)
    above_fc = bars[f"{metric.upper()}_SIG_VS_FC"].eq(True)
    return ["*" if significant else "" for significant in above_fc]


def _row_label(row, show_norm, show_extraction):
    """Row label: the feature, plus normalization / extraction when they vary.

    The model is identified by bar color and the experiment by hatching, so
    neither appears in the label.
    """
    parts = [row["Feature"]]
    # T1 isn't fMRI data, so normalization / timeseries extraction don't apply.
    if row["Feature"] != T1:
        if show_norm:
            norm = row["Voxelwise Normalization"]
            parts.append(NORMALIZATION_LABELS.get(norm, norm))
        if show_extraction:
            parts.append(row["Timeseries Extraction"])
    return " · ".join(parts)


def _dedupe_by_label(rows, metric, label_fn):
    """Reduce rows of one model + experiment that share a label (e.g. T1 under
    different normalizations) to the one with the highest `metric`."""
    keys = rows.apply(
        lambda row: (row["Foundation Model"], row["Experiment"], label_fn(row)), axis=1
    )
    best = rows.groupby(keys)[metric].idxmax()
    return rows.loc[best].reset_index(drop=True)


@dataclass
class TargetRanking:
    """The bars and reference values of one target's figure."""

    bars: pd.DataFrame  # foundation-model rows, best first
    labels: list[str]  # row label of each bar
    fc: pd.Series  # functional connectivity baseline (drawn as a line + CI band)
    dummy: pd.Series  # dummy classifier (chance level)
    model_rows: pd.DataFrame  # every foundation-model row before de-duplication


def rank_target(target_rows: pd.DataFrame, target: str) -> TargetRanking:
    """Rank one target's foundation-model rows by its sort metric."""
    model_rows = target_rows[~target_rows["Experiment"].isin(REFERENCE_EXPERIMENTS)]
    reference = target_rows[target_rows["Experiment"].isin(REFERENCE_EXPERIMENTS)]
    dummy = reference[reference["Feature"] == DUMMY].iloc[0]

    # Functional connectivity is atlas-independent, so its rows must agree.
    fc_rows = reference[reference["Feature"] == FC]
    metric_keys = [key for key, _ in METRICS]
    if not np.allclose(fc_rows[metric_keys], fc_rows[metric_keys].iloc[0]):
        raise ValueError(f"Functional connectivity baseline differs by atlas for {target}")

    # Labels only spell out normalization / extraction if a model varies in it;
    # T1 rows carry "N/A" for both and would otherwise look like variation.
    per_model = model_rows[model_rows["Feature"] != T1].groupby("Foundation Model")
    label_fn = partial(
        _row_label,
        show_norm=per_model["Voxelwise Normalization"].nunique().max() > 1,
        show_extraction=per_model["Timeseries Extraction"].nunique().max() > 1,
    )

    sort_metric = TARGET_SORT_METRIC.get(target, DEFAULT_SORT_METRIC)
    deduped = _dedupe_by_label(model_rows, sort_metric, label_fn)
    bars = deduped.sort_values(sort_metric, ascending=False).reset_index(drop=True)
    labels = [label_fn(row) for _, row in bars.iterrows()]
    return TargetRanking(bars, labels, fc_rows.iloc[0], dummy, model_rows)
