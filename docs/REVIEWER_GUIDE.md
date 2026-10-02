# Reviewer guide

The assignment requires a trained linear model, conversion into an MoE, and
proof that training continues and loss falls. The submission uses an actual
affine classifier and a small real dataset so every stage can be reproduced
on a CPU.

## Requirement-to-evidence map

| Requirement | Implementation | Measured evidence |
|---|---|---|
| Train a linear model | `Linear` in [train.py](../train.py): `x @ W + b`; no hidden layer | [All 15 cases](EXPERIMENT.md#every-assignment-case), pretrained checkpoints and epoch CSVs |
| Convert its trained weights | `MoE` makes independent copies of trained `W,b`; top-2 gates mix logits | Conversion error below 1e−12 in every case; converted checkpoints |
| Continue MoE training | Task/auxiliary gradients, sparse dispatch and Adam | Train and validation CE endpoints fall in every case |
| Show that experts and router learn | Different assignments produce different updates | Nonzero expert/router weight changes, [class routing, entropy and gradients](../results/router_diagnostics.png) |
| Compare fairly | Equal-step linear, projection-budget linear, scratch MoE and no-balance ablation | [All branch comparisons](../results/comparison.csv), paired and per-split statistics |
| Reproduce the work | Five seeds × three saved stratified splits, pinned dependencies, isolated fresh runs | [Raw summaries](../results/summary.json), checkpoints, checksums and CI |
| Supply an executable notebook | Embedded source definitions, fresh nbclient/ipykernel execution | [Executed notebook](../linear_to_moe.ipynb), [execution provenance](../results/notebook_execution.json) |
| Continue from saved state | Model, Adam moments/step and shuffle RNG are restored | Exact-next-update check and an actual ten-epoch continuation in the notebook |

## Review in five minutes

1. Read the measured results in [README](../README.md).
2. Open the [executed notebook](../linear_to_moe.ipynb): per-case losses,
   aggregate comparisons, five embedded plots, and checkpoint continuation.
3. Inspect the [full report](EXPERIMENT.md), especially all cases, paired
   comparisons, budget accounting and limitations.
4. Run `python check_submission.py` to independently validate saved evidence.

## Full reproduction

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

## Interpret the evidence correctly

Loss reduction is an endpoint comparison, not a promise of monotonic loss at
every epoch. The MoE has more active parameters than the starting classifier.
Projection MAC matching excludes backward, optimizer and gate costs. Accuracy
gains vary; balancing is not consistently better. The three splits reuse the
same dataset, so their variation is descriptive rather than an iid significance
test. The model is a literal linear classifier upcycled into affine experts,
not a transformer or language model. All baselines and unfavorable results are
retained. AI assistance is disclosed in the README and report.
