"""Styling shared by every figure in this repository."""

import contextlib
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

FONT_FAMILY = "sans-serif"
FONT_SANS = ["Inter", "Liberation Sans", "DejaVu Sans"]


@contextlib.contextmanager
def figure_style(font_scale: float):
    """Seaborn's whitegrid theme with the project fonts, at `font_scale`."""
    with (
        sns.axes_style(
            "whitegrid",
            {
                "grid.color": ".9",
                "font.family": FONT_FAMILY,
                "font.sans-serif": FONT_SANS,
            },
        ),
        sns.plotting_context("notebook", font_scale=font_scale),
    ):
        yield


def save_figure(fig, path: Path, dpi: int) -> None:
    """Save `fig` cropped to its content, then close it."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
