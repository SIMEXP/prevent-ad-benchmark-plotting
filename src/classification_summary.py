"""Downstream classification figures, drawn from summary_classification.tsv.

Three figure types (see `main` for the files written):

  - `plot_classification_summary`: one figure per target, with an accuracy, AUC
    and precision panel. Rows are the foundation-model features, ranked once
    (best first) so a row can be followed across panels.
  - `plot_accuracy_comparison`: accuracy panels of several targets side by side.
  - `plot_baseline_comparison`: the non-foundation-model baselines across targets.

How a foundation-model bar is drawn:
  - color = foundation model; hatched = fine-tuned ("transfer"), plain = frozen
    ("adaptation")
  - faded = its mean is below the dummy classifier's chance level
  - whisker = 95% CI of the mean over splits
  - "*" = significantly above the functional connectivity baseline (only shown
    when that baseline itself beats chance)

Reference marks: the functional connectivity baseline is a solid red line with a
shaded 95% CI band, and the dummy classifier's chance level a dashed red line.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from plot_style import figure_style, save_figure
from summary_data import (
    DUMMY,
    FC,
    METRICS,
    REFERENCE_EXPERIMENTS,
    load_summary,
    prepare_summary,
    rank_target,
    significance_marks,
)

COLOR_DUMMY = "#d62728"  # dummy classifier: dashed line
COLOR_FC_LINE = "#d62728"  # functional connectivity mean: solid line
COLOR_FC = "#2a78d6"  # functional connectivity CI band, and its bar in the baseline figure
COLOR_TIMESERIES = "#9467bd"
MODEL_COLORS = {"brainharmonix": "#eda100", "brainlm650M": "#1baf7a"}

HATCH_TRANSFER = "////"
BELOW_CHANCE_ALPHA = 0.5
CI_BAND_ALPHA = 0.15
BAR_HEIGHT = 0.5

# Seaborn font scale for every figure. Text that has to fit is sized in points
# with FontSizes instead, so this only affects what isn't set explicitly.
SEABORN_FONT_SCALE = 0.7


@dataclass(frozen=True)
class FontSizes:
    """Font sizes in points."""

    body: float  # row labels, legend
    title: float  # panel and figure titles
    tick: float  # x tick labels
    axis_label: float  # x axis label


# tick / axis_label: seaborn's notebook defaults at SEABORN_FONT_SCALE
COMPACT_FONTS = FontSizes(
    body=6, title=9, tick=11 * SEABORN_FONT_SCALE, axis_label=12 * SEABORN_FONT_SCALE
)
COMFORTABLE_FONTS = FontSizes(body=7, title=10.5, tick=7, axis_label=8)
BASELINE_FONTS = FontSizes(body=6, title=9, tick=6, axis_label=8)

# Figure widths in inches.
PER_TARGET_WIDTH_IN = 7.0  # all panels of one target figure
COMPARISON_PANEL_WIDTH_IN = 3.5
BASELINE_PANEL_WIDTH_IN = 7.5 / 2.54


# --- Shared drawing helpers -------------------------------------------------


def _bars_figure_height(n_bars, extra):
    """Height in inches of a figure with `n_bars` rows plus `extra` for title/legend."""
    return max(0.28 * n_bars, 4) + extra


def _error_lengths(bars, key):
    """Whisker lengths below and above the mean (0 where the CI is missing)."""
    low = (bars[key] - bars[f"{key}_low"]).fillna(0.0)
    high = (bars[f"{key}_high"] - bars[key]).fillna(0.0)
    return low, high


def _draw_error_bars(ax, positions, values, err_low, err_high, capsize=2.5):
    ax.errorbar(
        values,
        positions,
        xerr=[err_low, err_high],
        fmt="none",
        ecolor=".3",
        elinewidth=1,
        capsize=capsize,
        capthick=1,
        zorder=3,
    )


def _bar_row_indices(ax):
    """(patch, row index) of every bar on `ax`.

    Bars are matched to rows by their y-position, not draw order: seaborn draws
    bars grouped by hue (color), not in top-to-bottom row order.
    """
    for patch in ax.patches:
        yield patch, round(patch.get_y() + patch.get_height() / 2)


def _style_bars(ax, ranking, key):
    """Hatch fine-tuned bars, and fade bars whose mean is below chance."""
    bars = ranking.bars
    for patch, row in _bar_row_indices(ax):
        if not 0 <= row < len(bars):
            continue
        if bars.iloc[row]["Experiment"] == "transfer":
            patch.set_hatch(HATCH_TRANSFER)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.5)
        if bars.iloc[row][key] < ranking.dummy[key]:
            patch.set_alpha(BELOW_CHANCE_ALPHA)


def _annotate_significance(ax, ranking, key, err_high, fontsize):
    marks = significance_marks(ranking.bars, key, ranking.fc, ranking.dummy)
    for row, (value, err, mark) in enumerate(zip(ranking.bars[key], err_high, marks)):
        if mark:
            ax.text(value + err + 0.01, row, mark, va="center", fontsize=fontsize)


def _draw_fc_reference(ax, fc, key):
    """The functional connectivity baseline: mean line over a shaded 95% CI band."""
    ax.axvspan(
        fc[f"{key}_low"], fc[f"{key}_high"], color=COLOR_FC, alpha=CI_BAND_ALPHA, zorder=1
    )
    ax.axvline(fc[key], color=COLOR_FC_LINE, lw=1.5, zorder=3)


def _decorate_bars(ax, ranking, key, fonts):
    """Everything drawn on top of the foundation-model bars of one metric panel."""
    _style_bars(ax, ranking, key)
    err_low, err_high = _error_lengths(ranking.bars, key)
    _draw_error_bars(ax, np.arange(len(ranking.bars)), ranking.bars[key], err_low, err_high)
    _annotate_significance(ax, ranking, key, err_high, fonts.body)
    _draw_fc_reference(ax, ranking.fc, key)
    # the dummy classifier never predicts the positive class, so its precision
    # is always 0 and a chance line would carry no information
    if key != "precision":
        ax.axvline(ranking.dummy[key], color=COLOR_DUMMY, ls="--", lw=1.5, zorder=3)


def _add_model_legend(fig, models, fontsize, anchor_y):
    """Two-row legend explaining the reference marks, models and experiments."""
    grey = {"facecolor": "lightgrey", "edgecolor": "black"}
    entries = [
        (plt.Line2D([0], [0], color=COLOR_FC_LINE, lw=1.5), "Functional connectivity mean"),
        (
            plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_FC, alpha=CI_BAND_ALPHA, edgecolor="none"),
            "Functional connectivity 95% CI",
        ),
        (
            plt.Line2D([0], [0], color=COLOR_DUMMY, ls="--", lw=1.5),
            "Dummy classifier (chance level)",
        ),
        *[(plt.Rectangle((0, 0), 1, 1, color=MODEL_COLORS[m]), m) for m in models],
        (plt.Rectangle((0, 0), 1, 1, **grey), "Adaptation (frozen)"),
        (plt.Rectangle((0, 0), 1, 1, hatch=HATCH_TRANSFER, **grey), "Transfer (fine-tuned)"),
    ]
    handles, labels = zip(*entries)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, anchor_y),
        ncol=(len(handles) + 1) // 2,
        frameon=False,
        fontsize=fontsize,
    )


def _model_names(rankings):
    return pd.concat([r.model_rows for r in rankings])["Foundation Model"].unique()


def _target_rankings(df, targets):
    return [rank_target(df[df["Target"] == t], t) for t in targets]


# --- Per-target figures -----------------------------------------------------


def _long_format(bars, bar_ids):
    """One row per (bar, metric), as seaborn's catplot expects."""
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "bar": bar_ids,
                    "model": bars["Foundation Model"],
                    "metric": title,
                    "value": bars[key],
                }
            )
            for key, title in METRICS
        ],
        ignore_index=True,
    )


def _draw_metric_panel(ax, key, title, ranking, fonts):
    """One metric panel of a per-target figure, on top of seaborn's bars."""
    _decorate_bars(ax, ranking, key, fonts)
    # zoom the precision panel in when every bar (with its whisker) is small
    _, err_high = _error_lengths(ranking.bars, key)
    fits_zoom = key == "precision" and (ranking.bars[key] + err_high).max() < 0.55
    ax.set_xlim(0, 0.6 if fits_zoom else 1.12)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    ax.tick_params(axis="x", labelsize=fonts.tick)
    ax.set_xlabel(title, fontsize=fonts.axis_label)
    ax.yaxis.grid(False)


def _plot_target(ranking, target, fonts):
    """The figure of one target: a panel per metric over shared, ranked rows."""
    n_bars = len(ranking.bars)
    bar_ids = [f"bar{i}" for i in range(n_bars)]  # row labels can repeat
    height = _bars_figure_height(n_bars, extra=1)
    panel_width_in = PER_TARGET_WIDTH_IN / len(METRICS)

    with figure_style(SEABORN_FONT_SCALE):
        grid = sns.catplot(
            data=_long_format(ranking.bars, bar_ids),
            x="value",
            y="bar",
            hue="model",
            col="metric",
            col_order=[title for _, title in METRICS],
            order=bar_ids,
            kind="bar",
            palette=MODEL_COLORS,
            dodge=False,
            width=BAR_HEIGHT,
            saturation=1,
            edgecolor="none",
            errorbar=None,
            sharex=False,
            sharey=True,
            height=height,
            aspect=panel_width_in / height,
            legend=False,
        )

    for key, title in METRICS:
        _draw_metric_panel(grid.axes_dict[title], key, title, ranking, fonts)

    grid.set_titles("")
    grid.set_ylabels("")
    grid.set_yticklabels(ranking.labels, fontsize=fonts.body)
    grid.despine(left=True)
    _add_model_legend(
        grid.figure, ranking.model_rows["Foundation Model"].unique(), fonts.body, anchor_y=1.0
    )
    grid.figure.suptitle(target, fontsize=fonts.title, fontweight="bold", y=1.04)
    grid.tight_layout()
    return grid.figure


def _target_filename(target):
    """"β-amyloid SUVR > 1.26" -> "beta_amyloid_suvr_1_26.png"."""
    slug = re.sub(r"\W+", "_", target.replace("β", "beta")).strip("_").lower()
    return f"{slug}.png"


def plot_classification_summary(df: pd.DataFrame, output_dir: Path):
    """Write one figure per target (all foundation models together) into output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = prepare_summary(df)

    for target, target_rows in df.groupby("Target"):
        ranking = rank_target(target_rows, target)
        figure = _plot_target(ranking, target, COMPACT_FONTS)
        save_figure(figure, output_dir / _target_filename(target), dpi=150)


# --- Accuracy comparison across targets ---------------------------------------


def _draw_accuracy_panel(ax, target, ranking, fonts):
    n_bars = len(ranking.bars)
    positions = np.arange(n_bars)
    colors = [MODEL_COLORS[model] for model in ranking.bars["Foundation Model"]]
    ax.barh(
        positions,
        ranking.bars["accuracy"],
        color=colors,
        edgecolor="none",
        height=BAR_HEIGHT,
        zorder=2,
    )
    _decorate_bars(ax, ranking, "accuracy", fonts)
    ax.set_yticks(positions)
    ax.set_yticklabels(ranking.labels, fontsize=fonts.body)
    ax.invert_yaxis()  # highest accuracy on top, as in the per-target figures
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Accuracy", fontsize=fonts.axis_label)
    ax.tick_params(axis="x", labelsize=fonts.tick)
    ax.set_title(target, fontsize=fonts.title)
    ax.yaxis.grid(False)


def plot_accuracy_comparison(
    df: pd.DataFrame,
    targets: list[str],
    output_dir: Path,
    filename: str = "accuracy_comparison.png",
    title: str | None = None,
):
    """One accuracy panel per target, side by side, each independently ranked."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rankings = _target_rankings(prepare_summary(df), targets)

    fonts = COMFORTABLE_FONTS
    height = _bars_figure_height(max(len(r.bars) for r in rankings), extra=1.3)
    with figure_style(SEABORN_FONT_SCALE):
        fig, axes = plt.subplots(
            1, len(targets), figsize=(COMPARISON_PANEL_WIDTH_IN * len(targets), height)
        )
        for ax, target, ranking in zip(np.atleast_1d(axes), targets, rankings):
            _draw_accuracy_panel(ax, target, ranking, fonts)

        sns.despine(fig, left=True)
        _add_model_legend(fig, _model_names(rankings), fonts.body, anchor_y=1.0)
        fig.suptitle(
            title or " vs. ".join(targets), fontsize=fonts.title, fontweight="bold", y=1.1
        )
        fig.tight_layout(pad=0.8)
        save_figure(fig, output_dir / filename, dpi=150)


# --- Baseline comparison across targets ---------------------------------------

TIMESERIES_FEATURE = "Timeseries, top 75 PCs"
TIMESERIES_ATLAS = "Schaefer400"
BASELINE_METRICS = METRICS[:2] + [("f1", "F1")] + METRICS[2:]  # Accuracy, AUC, F1, Precision
BASELINE_GRID = (2, 2)  # rows, columns of metric panels
BASELINE_BAR_OFFSET = 0.16  # the two bars of a target sit this far above / below its row
BASELINE_BAR_HEIGHT = BASELINE_BAR_OFFSET * 1.8


class BaselineRows(NamedTuple):
    """One target's non-foundation-model reference rows."""

    fc: pd.Series
    timeseries: pd.Series
    dummy: pd.Series


def _baseline_rows(df, targets):
    rows = {}
    for target in targets:
        reference = df[(df["Target"] == target) & df["Experiment"].isin(REFERENCE_EXPERIMENTS)]
        rows[target] = BaselineRows(
            fc=reference[reference["Feature"] == FC].iloc[0],
            timeseries=reference[
                (reference["Feature"] == TIMESERIES_FEATURE)
                & (reference["Atlas"] == TIMESERIES_ATLAS)
            ].iloc[0],
            dummy=reference[reference["Feature"] == DUMMY].iloc[0],
        )
    return rows


def _draw_baseline_panel(ax, key, title, targets, rows, fonts, show_target_labels):
    """One metric panel: an FC bar and a timeseries bar per target."""
    positions = np.arange(len(targets))

    # Chance level of the dummy classifier is a constant for AUC (0.5) and for
    # F1 / precision (0: it never predicts the positive class), so one line
    # stands in for it there. For accuracy it varies by target and is drawn
    # per target below.
    if key == "auc":
        ax.axvline(0.5, color=COLOR_DUMMY, ls="--", lw=1.2, zorder=2)
    elif key in ("f1", "precision"):
        ax.axvline(0.0, color=COLOR_DUMMY, ls="--", lw=1.2, zorder=2)

    for series, offset, color in (
        ("fc", -BASELINE_BAR_OFFSET, COLOR_FC),
        ("timeseries", BASELINE_BAR_OFFSET, COLOR_TIMESERIES),
    ):
        series_rows = pd.DataFrame([getattr(rows[t], series) for t in targets])
        err_low, err_high = _error_lengths(series_rows, key)
        ax.barh(
            positions + offset,
            series_rows[key],
            color=color,
            edgecolor="none",
            height=BASELINE_BAR_HEIGHT,
            zorder=2,
        )
        _draw_error_bars(ax, positions + offset, series_rows[key], err_low, err_high, capsize=2)

    if key == "accuracy":
        for position, target in zip(positions, targets):
            chance = rows[target].dummy[key]
            ax.plot(
                [chance, chance],
                [position - 0.4, position + 0.4],
                color=COLOR_DUMMY,
                ls="--",
                lw=1.2,
                zorder=3,
            )

    ax.set_yticks(positions)
    ax.set_yticklabels(targets if show_target_labels else [], fontsize=fonts.body)
    ax.set_ylim(len(targets) - 0.5, -0.5)  # first target on top
    ax.set_xlim(0, 1.0)
    ax.tick_params(axis="x", labelsize=fonts.tick)
    ax.set_title(title, fontsize=fonts.title, loc="left")
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, alpha=0.5)


def plot_baseline_comparison(
    df: pd.DataFrame,
    targets: list[str],
    output_dir: Path,
    filename: str = "baseline_comparison.png",
    title: str | None = None,
):
    """One panel per metric (a 2x2 grid), comparing across targets the
    non-foundation-model baselines: functional connectivity and the top-75-PC
    timeseries classifier (Schaefer400 atlas only), against the dummy
    classifier's chance level."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _baseline_rows(prepare_summary(df), targets)

    fonts = BASELINE_FONTS
    grid_rows, grid_cols = BASELINE_GRID
    width_in = BASELINE_PANEL_WIDTH_IN * grid_cols
    height_in = grid_rows * (0.26 * len(targets) + 0.35) + 0.7

    with figure_style(SEABORN_FONT_SCALE):
        fig, axes = plt.subplots(
            grid_rows, grid_cols, figsize=(width_in, height_in), squeeze=False
        )
        for i, (key, metric_title) in enumerate(BASELINE_METRICS):
            grid_row, grid_col = divmod(i, grid_cols)
            _draw_baseline_panel(
                axes[grid_row][grid_col],
                key,
                metric_title,
                targets,
                rows,
                fonts,
                show_target_labels=grid_col == 0,
            )

        sns.despine(fig, left=True)
        fig.legend(
            [plt.Rectangle((0, 0), 1, 1, color=color) for color in (COLOR_FC, COLOR_TIMESERIES)]
            + [plt.Line2D([0], [0], color=COLOR_DUMMY, ls="--", lw=1.2)],
            ["Functional connectivity", TIMESERIES_FEATURE, "Dummy classifier (chance level)"],
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=3,
            frameon=False,
            fontsize=fonts.body,
        )
        fig.suptitle(
            title or " vs. ".join(targets), fontsize=fonts.title, fontweight="bold", y=0.995
        )
        fig.tight_layout(pad=0.2, h_pad=0.3, w_pad=0.3, rect=(0, 0, 1, 0.93))
        save_figure(fig, output_dir / filename, dpi=150)


def main():
    summary = load_summary(Path("outputs/downstream-summaries/summary_classification.tsv"))
    output_dir = Path("outputs/downstream-summaries/plots")

    plot_classification_summary(summary, output_dir)
    plot_accuracy_comparison(
        summary,
        ["Sex", "Age (binary)"],
        output_dir,
        filename="sex_age_accuracy.png",
        title="Sex and age prediction validates that the foundation model\n"
        "features carry meaningful but weak signals",
    )
    plot_baseline_comparison(
        summary,
        ["Sex", "Age (binary)", "MCI status", "β-amyloid SUVR > 1.26"],
        output_dir,
        filename="baseline_comparison.png",
        title="Non-foundation-model baselines across targets",
    )


if __name__ == "__main__":
    main()
