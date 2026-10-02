# Working with the experiment

Use Python 3.12 and `python -m pip install -r requirements-dev.txt`.
Keep the submitted evidence intact when experimenting:

```bash
python train.py --out runs/my-experiment
python -c "from plots import plot_results; plot_results('runs/my-experiment')"
```

Initialization seeds, split seeds, epochs, expert count, top-k, batch size and
balance coefficient are CLI options. All branches use the recorded protocol.
Never select seeds or alter stopping rules using test results.

For an intentional new submission, first execute `train.py` into a separate
output directory. Review all results before replacing `results/`. Then rebuild
the genuine Jupyter notebook and measured documentation:

```bash
python plots.py
python build_artifacts.py
python scripts/update_manifest.py
python check_submission.py --reproduce --check-singleton --execute-notebook
```

The builder executes in a temporary directory and compares numerical evidence
within explicit tolerances. It does not replace the submitted checkpoints.
Review generated README/report text and plots before committing. Update the
banner, citation version and description if the protocol changes.

Keep claims specific to the dataset and budget. Distinguish task CE from the
auxiliary objective, forward projection MACs from full FLOPs, and sample SD
from a confidence interval. Preserve all branches, seeds and unfavorable results.
