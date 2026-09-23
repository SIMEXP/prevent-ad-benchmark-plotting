# prevent-ad-benchmark-plotting

Lightweight plotting and reporting code for the PREVENT-AD foundation-model
benchmark (BrainHarmony and BrainLM). It turns the outputs of the
[prevent-ad-benchmark](../prevent-ad-benchmark) project into learning-curve figures,
a downstream-task summary table, and per-target comparison plots.

## Setup

Requires Python 3.14 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

`data/` is a symlink to `../prevent-ad-benchmark/outputs`. Recreate it if the
projects live somewhere else:

```bash
ln -sfn /path/to/prevent-ad-benchmark/outputs data
```

## Pipeline

Run everything from the repository root.
Step 3 needs step 2 to have run first. Step 1 is independent.

### 1. Learning curves: [src/learning_curves.py](src/learning_curves.py)

Reads `data/finetune/{model}/{condition}/{split}/config.json`, which hold the
per-epoch `train_loss` and `val_loss` of each split.

### 2. Downstream summary: [src/downstream_stats.py](src/downstream_stats.py)

Loads the classification results of the four folders in `data/downstreams/`
(`baseline.brainharmonix`, `baseline.brainlm`, `brainharmonix`, `brainlm`),
skips SVM classifiers and regression targets, and writes one row per
model / experiment / normalization / feature / target combination with the mean and
95% CI of accuracy, AUC, F1 and precision over the splits.

Experiments:

- `baseline`: functional connectivity and timeseries features
- `chance`: dummy classifier
- `adaptation`: frozen foundation model
- `transfer`: finetuned foundation model

Statistical tests, one-sided (greater):

- **vs functional connectivity** (`*_VS_FC`, accuracy, AUC and precision): Welch's
  t-test against the baseline of the same atlas. Unpaired, because baseline and
  foundation-model results come from different splits.
- **vs the dummy classifier** (`*_VS_DUMMY`, accuracy and AUC): one-sample t-test
  against the dummy classifier's mean. Precision is not tested, since the dummy
  classifier's precision is always 0.
- `*_SIG_VS_*` means p < 0.05 (no correction for multiple comparisons).

Baseline and chance rows are not tested against themselves.

### 3. Downstream plots: [src/classification_summary.py](src/classification_summary.py)

Only the model variants listed in `SOUND_VARIATIONS` in
[src/summary_data.py](src/summary_data.py) are plotted: the ones whose learning
curves converge cleanly (brainharmonix with global normalization, brainlm650M
without global normalization and with gigaconnectome timeseries). Edit that set to
include others. Which figures are written, and their titles, are set in `main()`.

## Layout

```
src/
  learning_curves.py        learning-curve figures
  downstream_stats.py       summary table with CIs and significance tests
  classification_summary.py downstream figures (per target, Sex/Age, baselines)
  summary_data.py           loads and ranks the summary table for those figures
  plot_style.py             theme and saving shared by all figures
data -> ../prevent-ad-benchmark/outputs   (symlink, inputs)
outputs/                                  (generated figures and tables)
```
