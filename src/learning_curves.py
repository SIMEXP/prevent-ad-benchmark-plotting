"""Learning curves of BrainLM and BrainHarmonix fine-tuning.

Faint lines are individual splits, bold lines the mean across splits, and the
grey strip under each panel shows how many splits reach each epoch (early
stopping means the right-hand end of a curve rests on few runs).
"""

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_style import figure_style, save_figure

BRAINHARMONIX_LABELS = {
    "zscore.self-supervised": "Global normalization",
    "nozscore.self-supervised": "No global normalization",
}
BRAINLM_LABELS = {  # run directory -> (normalisation, timeseries extraction)
    "zscore_brainlm.650M.selfsupervised": ("Global normalization", "BrainLM"),
    "zscore_gigaconnectome.650M.selfsupervised": ("Global normalization", "gigaconnectome"),
    "nozscore_brainlm.650M.selfsupervised": ("No global normalization", "BrainLM"),
    "nozscore_gigaconnectome.650M.selfsupervised": (
        "No global normalization",
        "gigaconnectome",
    ),
}
MODEL_DISPLAY_NAMES = {"brainlm.650M": "BrainLM-650M", "brainharmonix": "BrainHarmonix"}

COLOR_TRAIN = "#1f77b4"
COLOR_VAL = "#ff7f0e"
COLOR_RUN = "#8a8983"
PALETTE = {"Training": COLOR_TRAIN, "Validation": COLOR_VAL}
NORMALISATIONS = ["Global normalization", "No global normalization"]
A4_WIDTH_IN = 8.27  # fits one printed A4 page's width; height is set from content
LOG_RATIO = 10.0  # log y-axis if a curve's mean loss spans more than this (max / min)


def _split_index(name: str) -> int:
    """Split number of a run directory named "3" or "split3"."""
    return int(name) if name.isdigit() else int(name.split("split")[-1])


def _run_labels(model: str, condition: str) -> tuple[str, str, str]:
    """(model, normalisation, timeseries extraction) of one fine-tuning run."""
    if model == "brainharmonix":
        return model, BRAINHARMONIX_LABELS[condition], "gigaconnectome"
    normalisation, extraction = BRAINLM_LABELS[condition]
    return "brainlm.650M", normalisation, extraction


def load_curves(finetune_dir: Path) -> pd.DataFrame:
    """Load per-epoch train/val loss for all splits and conditions.

    Expected layout::

        finetune_dir/
          {model}/
            {condition}/
              {split_idx}/
                config.json   <- has "metrics": [{epoch, train_loss, val_loss}, ...]

    Returns a long-form DataFrame with columns:
        model, normalisation, timeseries_extraction, split, epoch, train_loss, val_loss
    """
    records = []
    for model_dir in sorted(Path(finetune_dir).iterdir()):
        for cond_dir in model_dir.iterdir():
            model, normalisation, extraction = _run_labels(model_dir.name, cond_dir.name)
            for split_dir in sorted(cond_dir.iterdir()):
                with open(split_dir / "config.json") as f:
                    metrics = json.load(f).get("metrics", [])
                for entry in metrics:
                    records.append(
                        {
                            "model": model,
                            "normalisation": normalisation,
                            "timeseries_extraction": extraction,
                            "split": _split_index(split_dir.name),
                            "epoch": entry["epoch"],
                            "train_loss": entry["train_loss"],
                            "val_loss": entry["val_loss"],
                        }
                    )
    return pd.DataFrame(records)


def _epoch_panel(spec, fig):
    """A loss axes with a short splits-per-epoch strip below it sharing the x axis."""
    inner = spec.subgridspec(2, 1, height_ratios=[5, 1], hspace=0.08)
    ax = fig.add_subplot(inner[0])
    ax_n = fig.add_subplot(inner[1], sharex=ax)
    return ax, ax_n


def _draw_condition(ax, ax_n, df, title, fontsize=9):
    """Draw one condition's curves in `ax` and its splits-per-epoch strip in `ax_n`."""
    long = df.melt(
        id_vars=["split", "epoch"],
        value_vars=["train_loss", "val_loss"],
        var_name="kind",
        value_name="loss",
    ).replace({"train_loss": "Training", "val_loss": "Validation"})
    kwargs = {"data": long, "x": "epoch", "y": "loss", "hue": "kind", "palette": PALETTE}

    sns.lineplot(
        **kwargs, units="split", estimator=None, alpha=0.12, lw=0.8, legend=False, ax=ax
    )
    sns.lineplot(
        **kwargs,
        style="kind",
        dashes={"Training": "", "Validation": (3, 2)},
        errorbar=None,
        lw=2,
        legend=False,
        ax=ax,
    )
    mean = long.groupby(["kind", "epoch"])["loss"].mean()
    if mean.max() / mean.min() > LOG_RATIO:
        ax.set_yscale("log")
    ax.set_title(title, fontsize=fontsize)
    ax.set_ylabel("Loss")
    ax.tick_params(labelbottom=False)
    ax.set_xlabel("")

    # rows per epoch = splits still running (each split has one row per epoch)
    sns.histplot(
        df,
        x="epoch",
        discrete=True,
        element="step",
        color="grey",
        alpha=0.35,
        linewidth=0,
        ax=ax_n,
    )
    ax_n.set_yticks([df.groupby("epoch")["split"].nunique().max()])
    ax_n.grid(False)
    ax_n.set_ylabel("Splits", fontsize=fontsize - 2)
    ax_n.tick_params(labelsize=fontsize - 2)
    ax_n.set_xlabel("Epoch")


def _legend_handles_and_labels():
    handles = [
        plt.Line2D([0], [0], color=COLOR_TRAIN, lw=2),
        plt.Line2D([0], [0], color=COLOR_VAL, lw=2, ls="--"),
        plt.Line2D([0], [0], color=COLOR_RUN, lw=1, alpha=0.5),
    ]
    return handles, ["Training (mean)", "Validation (mean)", "Individual splits"]


@dataclass(frozen=True)
class GridStyle:
    """Layout and fonts of the grid figure."""

    panel_fontsize: int
    title_fontsize: int
    legend_fontsize: int
    hspace: float
    wspace: float
    dpi: int
    font_scale: float | None  # seaborn font scale to apply, or None to keep the ambient one
    two_line_titles: bool  # a narrow panel can't fit a one-line title
    page_width_in: float | None  # fixed figure width, or None to size it from the panels

    def layout(self, n_rows: int, n_cols: int):
        """(figure size in inches, top edge of the panel grid as a figure fraction)."""
        if self.page_width_in is None:
            return (6 * n_cols, 4.2 * n_rows), 0.88
        row_height_in = 3.0  # tuned for legibility at A4 width, not full A4 height
        top_margin_in = 1.3  # reserved for suptitle + legend
        height_in = n_rows * row_height_in + top_margin_in
        return (self.page_width_in, height_in), 1 - top_margin_in / height_in


SCREEN_GRID = GridStyle(
    panel_fontsize=9,
    title_fontsize=14,
    legend_fontsize=9,
    hspace=0.35,
    wspace=0.3,
    dpi=150,
    font_scale=None,
    two_line_titles=False,
    page_width_in=None,
)
# Fonts are scaled up to stay legible at print width.
PRINT_GRID = GridStyle(
    panel_fontsize=11,
    title_fontsize=16,
    legend_fontsize=11,
    hspace=0.75,
    wspace=0.4,
    dpi=300,
    font_scale=1.3,
    two_line_titles=True,
    page_width_in=A4_WIDTH_IN,
)


def _draw_grid(conditions, output_file, style: GridStyle, transpose=False):
    """One row per normalisation and one column per (model, timeseries extraction),
    or the transpose of that when `transpose=True`. Panels are labelled a., b., ...
    in reading order."""
    columns = sorted({(model, extraction) for model, _, extraction in conditions})
    n_rows, n_cols = (
        (len(columns), len(NORMALISATIONS))
        if transpose
        else (len(NORMALISATIONS), len(columns))
    )
    figsize, top = style.layout(n_rows, n_cols)
    font_context = (
        sns.plotting_context("notebook", font_scale=style.font_scale)
        if style.font_scale
        else contextlib.nullcontext()
    )

    with font_context:
        fig = plt.figure(figsize=figsize)
        grid = fig.add_gridspec(
            n_rows, n_cols, hspace=style.hspace, wspace=style.wspace, top=top
        )
        for i, normalisation in enumerate(NORMALISATIONS):
            for j, (model, extraction) in enumerate(columns):
                row, col = (j, i) if transpose else (i, j)
                ax, ax_n = _epoch_panel(grid[row, col], fig)
                title = (
                    f"{model} · {extraction}\n{normalisation}"
                    if style.two_line_titles
                    else f"{model} · {normalisation} · {extraction}"
                )
                condition = conditions.get((model, normalisation, extraction))
                _draw_condition(ax, ax_n, condition, title, style.panel_fontsize)
                ax.text(
                    -0.12,
                    1.15,
                    chr(ord("a") + row * n_cols + col) + ".",
                    transform=ax.transAxes,
                    fontsize=style.panel_fontsize + 2,
                    fontweight="bold",
                    va="bottom",
                    ha="left",
                )
        fig.suptitle("Fine Tuning Learning Curves", y=1.03, fontsize=style.title_fontsize)
        fig.legend(
            *_legend_handles_and_labels(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=3,
            frameon=False,
            fontsize=style.legend_fontsize,
        )
        sns.despine(fig)
        save_figure(fig, output_file, dpi=style.dpi)


def _condition_filename(label) -> str:
    """("BrainLM-650M", "Global normalization", "BrainLM") -> "brainlm-650m_global-normalization_brainlm.png"."""
    parts = [str(part).lower().replace(" ", "-").replace(".", "-") for part in label]
    return "_".join(parts) + ".png"


def _draw_single_condition(df, label, output_file):
    fig = plt.figure(figsize=(6, 4.5))
    ax, ax_n = _epoch_panel(fig.add_gridspec(1, 1)[0], fig)
    _draw_condition(ax, ax_n, df, " · ".join(label))
    ax.legend(*_legend_handles_and_labels(), fontsize=7, loc="upper right")
    sns.despine(fig)
    save_figure(fig, output_file, dpi=150)


def plot_learning_curves(curve_df: pd.DataFrame, output_path: Path) -> None:
    """Plot train/val loss vs epoch for every split, per condition and as grids.

    A log y-axis is used when a condition's mean loss spans more than `LOG_RATIO`
    (max / min).

    Writes one PNG per (model, normalisation, timeseries extraction), plus
    `learning-curves.png` (a grid of all conditions) and
    `learning-curves-transposed.png` (the same at A4 print width).
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    keys = ["model", "normalisation", "timeseries_extraction"]
    conditions = {label: df for label, df in curve_df.groupby(keys)}

    with figure_style(font_scale=0.8):
        for label, df in conditions.items():
            _draw_single_condition(df, label, output_path / _condition_filename(label))
        _draw_grid(conditions, output_path / "learning-curves.png", SCREEN_GRID)
        _draw_grid(
            conditions,
            output_path / "learning-curves-transposed.png",
            PRINT_GRID,
            transpose=True,
        )


def main():
    output_path = Path("outputs/learning-curves")
    output_path.mkdir(parents=True, exist_ok=True)

    curves = load_curves(Path("data/finetune"))
    curves.to_csv(output_path / "learning-curve.tsv", sep="\t")

    curves["model"] = curves["model"].replace(MODEL_DISPLAY_NAMES)
    plot_learning_curves(curves, output_path)


if __name__ == "__main__":
    main()
