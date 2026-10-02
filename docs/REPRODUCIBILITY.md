# Reproducibility guide

Reproduce the affine-to-MoE experiment on a CPU, inspect the saved training
histories and checkpoints, and continue training from a saved model.

## Explore the experiment

1. Read the measured results in [README](../README.md).
2. Open the [executed notebook](../linear_to_moe.ipynb): per-case losses,
   aggregate comparisons, five embedded plots, and checkpoint continuation.
3. Inspect the [full report](EXPERIMENT.md), especially all cases, paired
   comparisons, budget accounting and limitations.
4. Run `python check_submission.py` to independently validate saved evidence.

## Reproduce the saved results

Use Python 3.12 from the repository root:

```bash
python -m pip install -r requirements-dev.txt
python check_submission.py --reproduce --check-singleton --execute-notebook
```

This audits all saved checkpoints and CSV endpoints, retrains every case in a
temporary directory, exercises singleton plotting, then clears the notebook
outputs and executes all its cells in a new kernel and empty directory.
It checks results with `rtol=1e-6`, `atol=1e-8`, excluding wall timings.
The same command runs in [GitHub Actions](../.github/workflows/verify.yml).

## Run a separate experiment

Save a new run in its own directory:

```bash
python train.py --out runs/my-experiment
python -c "from plots import plot_results; plot_results('runs/my-experiment')"
```

For a quick run, add `--seeds 42 --split-seeds 2026`. The full protocol uses
five initialization seeds and three stratified splits.

## Inspect the saved data

[results/](../results/) contains the original training log, per-epoch CSVs,
split indices, model checkpoints and routing diagnostics. Each final checkpoint
includes Adam moments, its update count and the shuffle RNG state.

The [report](EXPERIMENT.md#results-for-every-case) lists every case, and
[comparison.csv](../results/comparison.csv) contains all five branches.
[Notebook execution provenance](../results/notebook_execution.json) records
the fresh-kernel execution. The notebook's final code cell reloads a saved
MoE and trains for ten additional epochs.

## Interpret the results

Loss reduction is an endpoint comparison, not a promise of monotonic loss at
every epoch. The MoE has more active parameters than the starting classifier.
Projection MAC matching excludes backward, optimizer and gate costs. Accuracy
gains vary; balancing is not consistently better. The three splits reuse the
same dataset, so their variation is descriptive rather than an iid significance
test. The model is a literal linear classifier upcycled into affine experts,
not a transformer or language model. All baselines and unfavorable results are
retained. AI assistance is disclosed in the README and report.
