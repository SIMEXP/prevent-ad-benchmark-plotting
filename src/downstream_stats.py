import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

FEATURE_NAMES = {
    "timeseries": "Timeseries, top 75 PCs",
    "connectivity": "Functional connectivity",
    "fmri_mean": "fMRI (mean-pooling)",
    "t1_mean": "T1 (mean-pooling)",
    "harmonizer_cls": "Harmonizer (CLS)",
    "harmonizer_latent_mean": "Harmonizer (latent tokens mean-pooling)",
    "cls_token": "Attention head CLS Token",
    "cls_embedding": "CLS Embedding",
    "mean_embedding": "Mean-pooling CLS Embedding",
    "max_embedding": "Max-pooling CLS Embedding",
}

TARGET_NAMES = {
    "sex": "Sex",
    "age": "Age (years)",
    "splifhalfage": "Age (binary)",
    "progess2mci": "MCI Progression",
    "centiloidbin": "Centiloid > 20",
    "centiloid": "Centiloid",
    "abSUVR": "β-amyloid SUVR",
    "abSUVRbin": "β-amyloid SUVR > 1.26",
}

# Reverse lookup: display name -> target key
_TARGET_NAMES_REVERSE = {v: k for k, v in TARGET_NAMES.items()}

# Renamed only in the final summary table's "Target" column. Kept separate from
# TARGET_NAMES because that dict's values must still match the literal "Target"
# text baked into the raw per-split result files, used by the reverse lookup
# above to recover the internal key in _extract_foundation_metadata.
TARGET_DISPLAY_OVERRIDES = {"MCI Progression": "MCI status"}


def _target_display(target):
    name = TARGET_NAMES.get(target, target)
    return TARGET_DISPLAY_OVERRIDES.get(name, name)


# Atlas mapping by model name
_ATLAS_MAP = {
    "brainharmonix": "Schaefer400",
    "brainlm": "A424",
    "brainlm650m": "A424",
}


def _extract_baseline_metadata(input_dir: Path, filepath: Path, row, idx):
    """Extract metadata from baseline result files (x-{feat}_y-{target}_{clf}_prediction.tsv)."""
    # Pattern: x-{feature}_y-{target}_{classifier}_prediction.tsv
    pattern = r"x-(.+)_y-(.+)_(svm|linear|dummy)_prediction\.tsv"
    match = re.match(pattern, filepath.name)
    if match.group(3) == "svm":
        return None

    source = input_dir.name

    if match.group(1) == "dummy":
        experiment = "chance"
        atlas = "N/A"
        voxelwise_normalization = "N/A"
        timeseries_extraction = "N/A"
    else:
        experiment, _ = source.split(".")
        atlas = _ATLAS_MAP[source.lower().split(".")[-1]]
        voxelwise_normalization = "yes"
        timeseries_extraction = "gigaconnectome"

    return {
        "feature": match.group(1),
        "target": match.group(2),
        "classifier": match.group(3),
        "foundation_model": "N/A",
        "experiment": experiment,
        "voxelwise_normalization": voxelwise_normalization,
        "timeseries_extraction": timeseries_extraction,
        "atlas": atlas,
        "split": idx,
    }


def _extract_foundation_metadata(input_dir: Path, filepath: Path, row, idx):
    """Extract metadata from foundation model result files."""
    match = re.match(r"(\w+)\.(\w+)\.(finetuned\.)?split(\d+)\.tsv", filepath.name)

    voxelwise_normalization_desc = match.group(1)
    if "_" not in voxelwise_normalization_desc:
        timeseries_extraction = "gigaconnectome"
    else:
        timeseries_extraction = voxelwise_normalization_desc.split("_")[-1]

    voxelwise_normalization = "no" if "no" in voxelwise_normalization_desc else "yes"
    # "transfer" = finetuned model, "adaptation" = frozen model
    experiment = "transfer" if match.group(3) else "adaptation"
    foundation_model = match.group(2)
    split_idx = int(match.group(4))
    atlas = _ATLAS_MAP[foundation_model.lower()]

    target_display = row["Target"]
    target_key = _TARGET_NAMES_REVERSE.get(target_display, target_display)
    feature = row["Features"]
    classifier = row["Classifier"].lower()

    if classifier == "svm":
        return None
    # the modality-specific encoders (T1, fMRI mean-pooling) are never finetuned,
    # so "transfer" rows for them just repeat the frozen ones
    if experiment == "transfer" and feature in ("t1_mean", "fmri_mean"):
        return None

    return {
        "feature": feature,
        "target": target_key,
        "classifier": classifier,
        "foundation_model": foundation_model,
        "experiment": experiment,
        "voxelwise_normalization": voxelwise_normalization,
        "timeseries_extraction": timeseries_extraction,
        "atlas": atlas,
        "split": split_idx,
    }


def load_results(input_dirs: list[Path]) -> pd.DataFrame:
    """Load all result files from multiple directories into a single DataFrame.

    Auto-detects format per directory: directories with *.split*.tsv files are
    foundation model results; others are baseline results.
    """
    records = []

    for input_dir in input_dirs:
        input_dir = Path(input_dir).resolve()
        is_foundation = any(input_dir.glob("*.split*.tsv"))
        extract_metadata = (
            _extract_foundation_metadata
            if is_foundation
            else _extract_baseline_metadata
        )
        files = sorted(input_dir.glob("*.split*.tsv" if is_foundation else "*.tsv"))
        print(f"  Found {len(files)} files in {input_dir.name}")

        for filepath in files:
            df = pd.read_csv(filepath, sep="\t", index_col=0)
            if "test_acc" not in df.columns:  # skipping regressions
                continue

            for idx, row in df.iterrows():
                metadata = extract_metadata(input_dir, filepath, row, idx)
                if metadata is None:  # e.g. SVM rows
                    continue

                records.append(
                    metadata
                    | {
                        "accuracy": row["test_acc"],
                        "auc": row["test_auc"],
                        "f1": row["test_f1"],
                        "precision": row["test_precision"],
                        "task_type": "classification",
                    }
                )

    return pd.DataFrame(records)


def _add_metric_with_ci(record, metric_name, values):
    """Add metric with 95% CI to record."""
    mean = values.mean()
    sd = values.std(ddof=1)
    n = len(values)
    if sd == 0:  # identical values across splits: the interval collapses to the mean
        ci_lower = ci_upper = mean
    else:
        ci_lower, ci_upper = stats.t.interval(
            0.95, df=n - 1, loc=mean, scale=sd / np.sqrt(n)
        )
    upper_name = metric_name.upper()
    record[upper_name] = mean
    record[f"{upper_name}_CI_LOW"] = ci_lower
    record[f"{upper_name}_CI_HIGH"] = ci_upper


TESTED_METRICS = ["accuracy", "auc", "precision"]
# The dummy classifier never predicts the positive class, so its precision is always 0
# and a test against it is meaningless.
DUMMY_TESTED_METRICS = ["accuracy", "auc"]


def _add_test(record, metric, ref_name, result):
    """Add T/DF/P/SIG columns for one t-test result (NaN if the test wasn't run)."""
    t_stat, dof, p_value = np.nan, np.nan, np.nan
    if result is not None:
        t_stat, dof, p_value = result.statistic, result.df, result.pvalue
    prefix = metric.upper()
    record[f"{prefix}_T_VS_{ref_name}"] = t_stat
    record[f"{prefix}_DF_VS_{ref_name}"] = dof
    record[f"{prefix}_P_VS_{ref_name}"] = p_value
    record[f"{prefix}_SIG_VS_{ref_name}"] = (
        (p_value < 0.05) if pd.notna(p_value) else np.nan
    )


def _get_baseline_group(baseline_df, feature, target, atlas):
    """Per-split rows of one baseline comparison group, or None if there are none.

    Restricted to `atlas` since the baseline experiments were run once per atlas
    (Schaefer400 for brainharmonix, A424 for brainlm).
    """
    matched = baseline_df[
        (baseline_df["feature"] == feature)
        & (baseline_df["atlas"] == atlas)
        & (baseline_df["target"] == target)
    ]
    return matched if not matched.empty else None


def _add_tests_vs_baselines(record, group, baseline_df, experiment, target, atlas):
    """Add one-sided t-tests against the FC baseline and the dummy classifier."""
    fc_group = dummy_group = None
    # baseline and chance rows are the references, so they aren't tested against themselves
    if experiment not in ("baseline", "chance"):
        fc_group = _get_baseline_group(baseline_df, "connectivity", target, atlas)
        dummy_group = _get_baseline_group(baseline_df, "dummy", target, "N/A")

    for metric in TESTED_METRICS:
        fc_test = None
        if fc_group is not None:  # distribution vs distribution
            fc_test = stats.ttest_ind(
                group[metric],
                fc_group[metric],
                equal_var=False,
                alternative="greater",
            )
        _add_test(record, metric, "FC", fc_test)

    for metric in DUMMY_TESTED_METRICS:
        dummy_test = None
        if dummy_group is not None:  # distribution vs reference value
            dummy_test = stats.ttest_1samp(
                group[metric], dummy_group[metric].mean(), alternative="greater"
            )
        _add_test(record, metric, "DUMMY", dummy_test)


GROUP_COLUMNS = [
    "foundation_model",
    "experiment",
    "voxelwise_normalization",
    "timeseries_extraction",
    "feature",
    "target",
    "classifier",
    "atlas",
]
METRICS = ["accuracy", "auc", "f1", "precision"]


def _summarize_group(group_key, group, baseline_df):
    """One summary-table row: metric means with CIs, and the tests against baselines."""
    voxelwise_normalization = group_key["voxelwise_normalization"]
    if "t1" in group_key["feature"]:
        voxelwise_normalization = "N/A"  # doesn't apply to T1

    record = {
        "Foundation Model": group_key["foundation_model"],
        "Atlas": group_key["atlas"],
        "Experiment": group_key["experiment"],
        "Voxelwise Normalization": voxelwise_normalization,
        "Timeseries Extraction": group_key["timeseries_extraction"],
        "Feature": FEATURE_NAMES.get(group_key["feature"], group_key["feature"]),
        "Target": _target_display(group_key["target"]),
        "Classifier": group_key["classifier"].upper(),
    }
    for metric in METRICS:
        _add_metric_with_ci(record, metric, group[metric])
    _add_tests_vs_baselines(
        record,
        group,
        baseline_df,
        group_key["experiment"],
        group_key["target"],
        group_key["atlas"],
    )
    return record


def make_summary_table(
    df: pd.DataFrame,
    output_dir: Path | None = None,
    baseline_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create summary table with mean and CI95% for all metrics, split into
    separate columns (`METRIC`, `METRIC_CI_LOW`, `METRIC_CI_HIGH`).

    Also runs one-sided (greater) t-tests, adding
    `METRIC_T_VS_*`/`METRIC_DF_VS_*`/`METRIC_P_VS_*`/`METRIC_SIG_VS_*` columns
    (SIG = p < 0.05):
      - accuracy, AUC and precision vs the functional-connectivity baseline of the
        same atlas (`*_VS_FC`): Welch's unpaired t-test, since baseline results come
        from independent CV folds while foundation-model results come from fixed
        train/test splits, so the samples are not matched.
      - accuracy and AUC vs the dummy classifier (`*_VS_DUMMY`): one-sample t-test
        against the dummy classifier's mean. Precision is not tested against it,
        since the dummy classifier's precision is always 0.
    Baseline and chance rows are not tested and get NaN in these columns.

    Args:
        df: results to summarize (from load_results).
        output_dir: if given, writes summary_classification.tsv here.
        baseline_df: results to compare against for the t-tests. Defaults to `df`
            itself, so callers that already include baseline rows in `df` (e.g.
            experiment='all' or 'baselines') don't need to pass anything extra;
            callers summarizing only a foundation model's own results (e.g.
            experiment='brainharmonix') should pass the baseline results here
            explicitly so the comparison has something to compare against.
    """
    if baseline_df is None:
        baseline_df = df

    summary_df = pd.DataFrame(
        [
            _summarize_group(dict(zip(GROUP_COLUMNS, group_key)), group, baseline_df)
            for group_key, group in df.groupby(GROUP_COLUMNS)
        ]
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        # foundation-model files also hold regression targets, which have no accuracy
        summary_df.dropna(subset=["ACCURACY"]).to_csv(
            output_dir / "summary_classification.tsv", index=False, sep="\t"
        )
    return summary_df


if __name__ == "__main__":
    input_dirs = [
        Path("data/downstreams/baseline.brainharmonix"),
        Path("data/downstreams/baseline.brainlm"),
        Path("data/downstreams/brainharmonix"),
        Path("data/downstreams/brainlm"),
    ]
    output_dir = Path("outputs/downstream-summaries")
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(input_dirs)
    df.to_csv(output_dir / "raw_classification_results.tsv", index=False, sep="\t")
    summary_df = make_summary_table(df, output_dir=output_dir)
