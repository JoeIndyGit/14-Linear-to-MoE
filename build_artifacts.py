"""Run a self-contained notebook in Jupyter and generate measured documentation.

The notebook runs in an empty temporary working directory. Its results are
compared with the committed script results using explicit float tolerances.
No saved training evidence is overwritten by notebook execution.
"""
import csv
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

import nbformat
from evidence import compare_evidence
from notebook_tools import execute_notebook

ROOT = Path(__file__).resolve().parent
REPO = "JoeIndyGit/14-Linear-to-MoE"
NAMES = {"linear": "Linear continuation", "moe": "Upcycled MoE",
         "linear_mac_matched": "Linear · projection budget", "scratch_moe": "MoE from scratch",
         "moe_no_balance": "Upcycled MoE · no balance"}


def source(name):
    s = (ROOT / name).read_text().split('if __name__ == "__main__":')[0]
    if name == "verify.py":
        s = re.sub(r"^from (?:train|evidence) import .+\n", "", s, flags=re.MULTILINE)
    if name == "plots.py":
        s = s.replace('matplotlib.use("Agg")\n', '')
    return s


def build_notebook(execute=True):
    cfg = json.loads((ROOT / "results/config.json").read_text())
    expected = json.loads((ROOT / "results/summary.json").read_text())
    n = len(expected)
    cells = []

    def md(text):
        cells.append(nbformat.v4.new_markdown_cell(text))

    def code(text, module=None):
        metadata = {"module": module} if module else {}
        cells.append(nbformat.v4.new_code_cell(text, metadata=metadata))

    md(f"""# Linear Becomes Experts

**An executed, reproducible linear → sparse MoE experiment**

Train a literal affine classifier on handwritten digits, copy its learned
weights into independent experts, then show that learning continues.
This notebook contains every implementation and runs without repository files.
The saved outputs were produced by a **fresh Jupyter kernel**, using nbclient.

**Run all** in Jupyter or Colab with a CPU. NumPy, scikit-learn, Matplotlib and
threadpoolctl are needed; they are usually already installed in Colab.
The digits data ships with scikit-learn. No GPU, API key or dataset download.

**Protocol:** {len(cfg['seeds'])} initialization seeds × {len(cfg['split_seeds'])} stratified
splits = {n} paired cases. All final checkpoints are fixed in advance.
Five branches compare equal steps, forward projection budgets, training from
scratch and removal of the router balancing loss. Variation across overlapping
splits is descriptive; it is not a statistical significance claim.
""")
    md(r"""## 1. Models and sparse dispatch

The starting classifier has no hidden layer: $z(x)=xW+b$.
Each copied expert starts with $W_e=W$ and $b_e=b$.
With normalized selected gates, $\sum_{e\in S(x)}g_e(x)=1$, so
$z_{MoE}(x)=\sum_{e\in S(x)}g_e(x)(xW_e+b_e)=z(x)$ at conversion.

Top-2 routing lets task gradients flow through selected softmax weights.
Hard expert selection is piecewise constant and is not differentiated.
The code evaluates an expert only on assigned input rows. All experts have
independent storage; no inputs are dropped. The scratch baseline uses the
same architecture with independently initialized untrained expert weights.
""")
    training = source("train.py")
    a, b = training.index("\n\nclass Adam:"), training.index("\n\ndef train_phase(")
    code(training[:a], "train.py")
    md("## 2. Adam, disjoint data splits and resumable checkpoints\n\nPixels are divided by the known maximum, 16. Split indices are saved. Checkpoints include model weights, Adam moments and the shuffle RNG state, and load without pickle.")
    code(training[a:b], "train.py")
    md(f"""## 3. Training protocol and comparisons

- Pretrain a linear model for {cfg['pretrain_epochs']} epochs, then fork the checkpoint.
- Continue the linear and upcycled MoE branches for {cfg['continuation_epochs']} epochs.
- The scratch MoE trains for the same total number of epochs; Adam resets at
  the same boundary. All branches share their corresponding batch orders.
- The projection-budget linear control receives enough extra epochs to match
  the MoE's cumulative affine projection MACs, including linear pretraining.
- Remove the balancing loss in a fifth branch while preserving its starting
  checkpoint and batches. No result is removed if an ablation wins.

The MAC budget excludes backward, optimizer, sort, softmax and mixing costs.
It is not a complete FLOP or wall-clock match. Test labels are used only for
fixed-final-checkpoint evaluation; train and validation are logged each epoch.
""")
    code(training[b:], "train.py")
    md("## 4. Numerical verification\n\nCheck all handwritten gradients by finite differences, the sparse dispatch reference, independent expert storage, exact resumption and tolerance behavior.")
    code(source("evidence.py"), "evidence.py")
    code(source("verify.py"), "verify.py")
    code('checks = verify()\nprint(json.dumps(checks, indent=2))\nPath("results").mkdir(exist_ok=True)\nPath("results/verification.json").write_text(json.dumps(checks, indent=2) + "\\n")\n')
    md(f"## 5. Execute all {n} paired cases\n\nThe final epoch is reported for every branch. No best checkpoint or best seed is selected.")
    kwargs = dict(seeds=tuple(cfg["seeds"]), split_seeds=tuple(cfg["split_seeds"]),
        pre_epochs=cfg["pretrain_epochs"], post_epochs=cfg["continuation_epochs"],
        experts=cfg["experts"], top_k=cfg["top_k"], batch_size=cfg["batch_size"], alpha=cfg["balance_alpha"])
    args = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    code(f'summaries, histories = run_experiment("results", verbose=False, {args})\n'
         'for s in summaries:\n'
         '    print(f"Split {s[\'split_seed\']}, seed {s[\'seed\']}: "\n'
         '          f"MoE train CE {s[\'moe_initial_train_ce\']:.6f} → {s[\'moe_final_train_ce\']:.6f}; "\n'
         '          f"validation {s[\'moe_initial_val_ce\']:.6f} → {s[\'moe_final_val_ce\']:.6f}")\n')
    md("## 6. Read the actual aggregate and paired comparisons\n\nCross-entropy is in nats and excludes the auxiliary loss. SD is a sample standard deviation, not a confidence interval. Paired deltas retain each case's seed and split.")
    code('''aggregate = json.loads(Path("results/aggregate.json").read_text())
print(f"{'Branch':<24} {'Train CE':>10} {'Val CE':>10} {'Test CE':>10} {'Test accuracy':>15}")
for name, branch in aggregate["branches"].items():
    print(f"{name:<24} {branch['train']['ce']['mean']:10.6f} {branch['val']['ce']['mean']:10.6f} "
          f"{branch['test']['ce']['mean']:10.6f} {100*branch['test']['accuracy']['mean']:14.3f}%")
print("\\nPaired MoE minus baseline (negative validation CE delta favors MoE):")
for name, values in aggregate["paired_comparisons"].items():
    v = values["val_ce"]
    print(name, "mean delta:", round(v["mean"], 6), "MoE wins:", v["moe_better"], "of", v["n"],
          "per-split mean deltas:", v["per_split_mean_delta"])
print("\\nMaximum conversion logit difference:", max(s["conversion_max_abs_logit_difference"] for s in summaries))
''')
    md("## 7. Inspect losses, baselines, projection budgets and the router\n\nPlots use the saved raw CSV/JSON data without smoothing. Router heatmaps show validation labels in the first configured case. Expert IDs are local to each run; a routing pattern is not a semantic expert label.")
    code(source("plots.py"), "plots.py")
    code('from IPython.display import display, Image\nfor path in plot_results("results"):\n    display(Image(filename=str(path)))\n')
    md("## 8. Continue training from a saved MoE\n\nReload the optimizer and shuffle RNG and train for ten additional epochs. This check uses an isolated restored model and does not change the fixed-final-epoch comparison above.")
    first = expected[0]
    code(f'''case = summaries[0]
checkpoint = Path("results") / f"split_{{case['split_seed']}}" / f"seed_{{case['seed']}}" / "moe_final.npz"
restored, optimizer, batch_rng, metadata = load_checkpoint(checkpoint)
data = get_data(case["split_seed"])
before = evaluate(restored, data["train"])
np.testing.assert_allclose(evaluate(restored, data["val"])["ce"], case["moe_final_val_ce"], rtol=1e-6, atol=1e-8)
previous_step = optimizer.t
with threadpool_limits(limits=1):
    resumed, optimizer, batch_rng = train_phase(restored, data, epochs=10, seed=0,
        phase="resumed_moe", global_offset=metadata["epoch"], lr=optimizer.lr,
        batch_size={cfg['batch_size']}, alpha={cfg['balance_alpha']}, router_lr=optimizer.router_lr,
        verbose=False, optimizer=optimizer, rng=batch_rng)
after = evaluate(restored, data["train"])
assert after["ce"] < before["ce"]
assert optimizer.t == previous_step + 10 * int(np.ceil(len(data["train"][1]) / {cfg['batch_size']}))
print("Restored model trained further:", before["ce"], "→", after["ce"])
print("Adam steps:", previous_step, "→", optimizer.t)
print("All checks passed.")
''')
    md("""## Interpretation and limits

The linear model learns, conversion preserves its predictions, and the
upcycled MoE keeps reducing training and validation loss. The baseline
comparisons show that the accuracy advantage varies by case and budget. The balancing
ablation does not consistently improve held-out performance; keep its full
results rather than assuming that balancing must help.

This is a small digit-classification experiment, not a transformer or language
model. Three overlapping data splits are not three independent datasets.
Top-k gradients are local to a fixed selection set. MAC accounting omits
several operations, and CPU timing is implementation-specific. There is no
inference-speed or universal-superiority claim.

References: [Sparse Upcycling](https://arxiv.org/abs/2212.05055),
[Switch Transformers](https://arxiv.org/abs/2101.03961),
[digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).
AI assistance was used to develop and document this experiment; outputs come
from actual executed training and numerical verification.
""")
    nb = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
        "language_info": {"name": "python", "version": cfg["python"]},
        "provenance": {"executor": "nbclient", "isolated_working_directory": True,
            "train_source_sha256": hashlib.sha256(source("train.py").encode()).hexdigest()}})
    if not execute:
        nb.metadata.provenance["execution_status"] = "pending"
        nb.cells[0].source = nb.cells[0].source.replace(
            "The saved outputs were produced by a **fresh Jupyter kernel**, using nbclient.",
            "**Execution status: prepared. CI will generate and verify fresh Jupyter outputs.**")
        nbformat.validate(nb)
        nbformat.write(nb, ROOT / "linear_to_moe.ipynb")
        print("Notebook prepared with no execution outputs.", flush=True)
        return
    print(f"Executing {n} cases in a fresh Jupyter kernel...", flush=True)
    with tempfile.TemporaryDirectory(prefix="linear-moe-notebook-build-") as folder:
        execute_notebook(nb, folder)
        actual = json.loads((Path(folder) / "results/summary.json").read_text())
        compare_evidence(actual, expected)
        shutil.copy(Path(folder) / "results/verification.json", ROOT / "results/verification.json")
    nb.metadata.provenance["execution_status"] = "completed"
    nbformat.write(nb, ROOT / "linear_to_moe.ipynb")
    info = {"executor": "nbclient", "nbclient": importlib.metadata.version("nbclient"),
        "nbformat": importlib.metadata.version("nbformat"), "ipykernel": importlib.metadata.version("ipykernel"),
        "fresh_kernel": True, "empty_working_directory": True, "schema_validation": "passed",
        "script_result_comparison": "passed", "rtol": 1e-6, "atol": 1e-8,
        "timing_metrics_excluded": True,
        "code_cells": sum(c.cell_type == "code" for c in nb.cells),
        "embedded_pngs": sum("image/png" in o.get("data", {}) for c in nb.cells
                             if c.cell_type == "code" for o in c.outputs)}
    if os.environ.get("GITHUB_ACTIONS") == "true":
        info.update(execution_environment="GitHub Actions", source_commit=os.environ.get("GITHUB_SHA"),
            workflow_run=f"https://github.com/{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{os.environ.get('GITHUB_RUN_ID')}")
    else:
        info["execution_environment"] = "local Jupyter"
    (ROOT / "results/notebook_execution.json").write_text(json.dumps(info, indent=2) + "\n")
    print("Fresh Jupyter notebook reproduced every script result within rtol=1e-6, atol=1e-8 (runtime excluded).", flush=True)


def write_report():
    cfg = json.loads((ROOT / "results/config.json").read_text())
    summaries = json.loads((ROOT / "results/summary.json").read_text())
    a = json.loads((ROOT / "results/aggregate.json").read_text())
    n = len(summaries)
    case = summaries[0]
    max_delta = max(s["conversion_max_abs_logit_difference"] for s in summaries)
    drop_train = 100 * (1 - a["moe_final_train_ce"]["mean"] / a["moe_initial_train_ce"]["mean"])
    drop_val = 100 * (1 - a["moe_final_val_ce"]["mean"] / a["moe_initial_val_ce"]["mean"])

    def fmt(v, scale=1, digits=6):
        return f"{scale*v['mean']:.{digits}f} ± {scale*v['std']:.{digits}f}"

    stage_table = "| Stage | Train CE ↓ | Validation CE ↓ |\n|---|---:|---:|\n"
    for label, prefix in (("Random linear", "initial"), ("Trained linear / conversion point", "linear_pretrain"),
                           ("Linear continuation", "linear_final"), ("Upcycled MoE", "moe_final")):
        stage_table += f"| {label} | {fmt(a[prefix+'_train_ce'])} | {fmt(a[prefix+'_val_ce'])} |\n"
    baseline_table = "| Final branch | Total epochs | Train CE | Validation CE | Test CE | Test accuracy |\n|---|---:|---:|---:|---:|---:|\n"
    for name, label in NAMES.items():
        m = a["branches"][name]
        baseline_table += f"| {label} | {case['branches'][name]['epochs']} | {fmt(m['train']['ce'])} | {fmt(m['val']['ce'])} | {fmt(m['test']['ce'])} | {fmt(m['test']['accuracy'],100,2)}% |\n"
    paired = "| Baseline | Mean paired Δ validation CE | MoE lower validation CE | Mean paired Δ test accuracy (pp) | MoE better / tie / worse accuracy |\n|---|---:|---:|---:|---:|\n"
    for name, values in a["paired_comparisons"].items():
        v, t = values["val_ce"], values["test_accuracy"]
        paired += f"| {NAMES[name]} | {fmt(v)} | {v['moe_better']}/{n} | {fmt(t,100,3)} | {t['moe_better']} / {t['ties']} / {n-t['moe_better']-t['ties']} |\n"
    per_split = "| Split seed | Linear validation CE | Upcycled MoE | Budget linear | Scratch MoE | No balance |\n|---|---:|---:|---:|---:|---:|\n"
    for split, values in a["per_split"].items():
        per_split += f"| {split} | " + " | ".join(f"{values[name]['val']['mean']:.6f}" for name in NAMES) + " |\n"
    all_runs = "| Split / seed | Pretrained train CE | MoE final train CE | Pretrained val CE | MoE final val CE | Linear test accuracy | MoE test accuracy |\n|---|---:|---:|---:|---:|---:|---:|\n"
    for s in summaries:
        all_runs += f"| {s['split_seed']} / {s['seed']} | {s['linear_pretrain_train_ce']:.6f} | {s['moe_final_train_ce']:.6f} | {s['linear_pretrain_val_ce']:.6f} | {s['moe_final_val_ce']:.6f} | {100*s['linear_final_test']['accuracy']:.2f}% | {100*s['moe_final_test']['accuracy']:.2f}% |\n"
    budget = "| Branch | Cumulative forward projection MACs | Relative to upcycled MoE |\n|---|---:|---:|\n"
    for name, label in NAMES.items():
        b = case["branches"][name]
        budget += f"| {label} | {b['forward_projection_macs']:,} | {b['forward_projection_macs']/case['branches']['moe']['forward_projection_macs']:.3f}× |\n"
    settings = f"""| Setting | Value |
|---|---|
| Dataset | 1,797 digits, 8×8 pixels, ten classes; `X / 16` |
| Each stratified split | 1,078 train / 359 validation / 360 test |
| Split seeds | {', '.join(map(str,cfg['split_seeds']))} |
| Initialization seeds | {', '.join(map(str,cfg['seeds']))} |
| Pretraining / continuation | {cfg['pretrain_epochs']} / {cfg['continuation_epochs']} epochs |
| Batch size / optimizer | {cfg['batch_size']} / handwritten Adam |
| Adam β₁, β₂, ε | 0.9, 0.999, 1e−8; no weight decay |
| Learning rates | 0.01 pretraining, 0.005 continuation, 0.002 router |
| Balancing coefficient | {cfg['balance_alpha']} (zero in ablation) |
| Experts / selected | {cfg['experts']} / {cfg['top_k']} |
| Device / precision | CPU / float64 / one BLAS thread |
| Reporting | Fixed final epoch in each branch, no early stopping |
"""
    method = r"""The literal linear classifier is $z=xW+b$, followed by a softmax for
classification. The MoE router computes $s=xR+r$, chooses a top-k set $S(x)$,
and uses $g_e=\exp(s_e)/\sum_{j\in S(x)}\exp(s_j)$ to mix selected expert logits.
No hidden layer is added to any expert.

At conversion $W_e=W$ and $b_e=b$, so normalized gates preserve the trained
logits. Each copy owns independent storage. Expert dispatch computes each
expert only on assigned rows; no capacity cap or dropped sample is used.

The task objective is ordinary mean cross-entropy. The training objective adds
$\alpha E\sum_e f_e P_e$, where $f_e$ is the fraction of top-k assignments
and $P_e$ is the mean full-router softmax probability. Assignment fractions
are stop-gradient, and gradients flow through full-router probabilities for
this auxiliary term. This adapts the Switch-style balancing form to top-k
assignment counts; it is not an implementation of Switch Transformer.

Task gradients flow through selected softmax weights within a fixed top-k
set. There is no derivative through the hard selection. At identical experts,
the task router gradient is zero to numerical precision; assignment-dependent
expert updates break symmetry. The balancing gradient can act from the start.
Finite differences test both task and combined objectives away from ties.
"""
    report = f"""# Experiment report

This experiment trains a literal affine classifier, copies its learned weights
into a sparse MoE, and measures continued learning. Across all **{n} paired
cases**, linear pretraining lowers loss, conversion preserves predictions,
and MoE continuation lowers both training and validation cross-entropy.

## Protocol

{settings}
All compared continuation branches start with reset Adam moments and the same
shuffle seed. The scratch MoE has independent random experts and receives the
same {cfg['pretrain_epochs']} + {cfg['continuation_epochs']} epoch schedule, including its Adam reset.
No learning rate or stopping rule was selected using the new test results.
The three new split definitions and all seeds were chosen before this expanded
suite. The pilot experiment used split 2026 and seeds 42–44; this expansion
is an audit extension, not a claim of wholly unseen experimental design.
Test labels are evaluated only at fixed-final checkpoints. Conversion checks
use all images without labels; router heatmaps use validation labels.

## Models and conversion

{method}
| Model | Stored parameters | Parameter entries used per image | Projection MACs per image |
|---|---:|---:|---:|
| Linear | {case['linear_params']} | {case['linear_params']} | {case['linear_projection_macs_per_sample']} |
| MoE | {case['moe_total_params']} | {case['moe_active_params_per_sample']} | {case['moe_projection_macs_per_sample']} |

Maximum conversion logit difference: **{max_delta:.3e}** across all cases and inputs.
Parameter entries are capacity counts, not FLOPs or a speed measurement.

## Continued learning

{stage_table}
Mean ± sample SD over {n} cases. CE is in nats and excludes balancing.
Accuracy standard deviations are in percentage points.
From conversion to the final MoE, mean train CE falls **{drop_train:.2f}%** and
mean validation CE falls **{drop_val:.2f}%**. These are endpoint reductions;
individual epoch curves can fluctuate.

![Measured loss curves](../results/loss_curves.png)

## Baselines and ablation

{baseline_table}
{paired}
Paired deltas are MoE minus baseline within the same seed and split. Negative
CE deltas favor MoE; positive accuracy deltas favor MoE. The MoE's accuracy
advantage varies. Removing balancing gives slightly better aggregate validation
CE and test accuracy here; balancing is not established as necessary for this
small dataset. The scratch MoE reaches a similar aggregate train loss, while
its validation CE is higher on average. These are observations from this suite,
not general claims about sparse upcycling.

![All baselines](../results/baseline_comparison.png)

### Variation across data splits

{per_split}
Each row averages {len(cfg['seeds'])} initialization seeds. The splits reuse images,
so their {n} case metrics are correlated. SD describes variation; we do not
report an iid standard error, confidence interval or significance test.

### Forward projection budget

For input dimension $d$, classes $C$, experts $E$ and selected experts $k$,
linear projections cost $dC$ MACs per image and MoE projections cost
$dE+kdC$. Biases, sorting, softmax, mixing, backward and optimizer work are
excluded. Both budgets include the shared linear pretraining. The linear budget
control runs **{case['branches']['linear_mac_matched']['continuation_epochs']} continuation epochs**, versus
{cfg['continuation_epochs']} for the MoE, so default cumulative projection costs match exactly.
For custom sizes the linear epoch budget rounds up and records the excess.

{budget}
The scratch baseline trains the MoE for all epochs and therefore has a higher
projection budget. No wall-clock or complete-training-FLOP equivalence is claimed.
`training_seconds` measures shuffling, gradients and Adam only; evaluation and
diagnostics are excluded. Timing is machine- and implementation-specific and
is excluded from reproduction comparisons.

![Projection-budget curves](../results/compute_budget.png)

## Router and expert diagnostics

Every copied expert changes weights; the router changes too. Final assignment
shares are nonzero in all cases. Independent expert copies finish different.
The diagnostic plots show class assignment shares, normalized full-router and
selected-gate entropy, and task/auxiliary router gradient norms at fixed probes.
Heatmaps use the first configured case, split {case['split_seed']} / seed {case['seed']}.
Expert IDs are permutation-symmetric across runs and are not averaged as
semantic identities. Routing patterns show conditional behavior but do not
prove that an expert represents a human-interpretable concept.

![Expert participation](../results/expert_routing.png)
![Class routing, entropy and gradients](../results/router_diagnostics.png)

## Results for every case

{all_runs}
Unrounded numbers and all five branches are in [comparison.csv](../results/comparison.csv)
and [summary.json](../results/summary.json). No seed is omitted.

## Reproducibility and verification

Use Python 3.12 and the pinned dependencies. `check_submission.py` checks saved
metrics by independently loading the checkpoints, reconstructing each split,
checking every epoch endpoint and recomputing the aggregates and diagnostics.
It also checks independent conversion copies, exact optimizer/RNG resumption,
finite-difference gradients and floating-point tolerance regressions.

```bash
python -m pip install -r requirements-dev.txt
python check_submission.py --reproduce --check-singleton --execute-notebook
```

The builder uses nbformat validation and nbclient with a fresh ipykernel in an
empty temporary directory. All source definitions are embedded in the notebook
and checked against the Python modules. Every code cell executes in order;
outputs include five real plots and a further ten-epoch checkpoint continuation.
The notebook must reproduce the script summaries with `rtol=1e-6`, `atol=1e-8`;
wall timings are excluded. Shape, list length and identifier changes still fail.
See [execution provenance](../results/notebook_execution.json) and
[numerical checks](../results/verification.json). GitHub Actions runs the same
saved-evidence, fresh-script, singleton-plot and fresh-Jupyter checks.

## Continue from a saved MoE

```python
from train import load_checkpoint, get_data, train_phase
model, optimizer, rng, meta = load_checkpoint(
    "results/split_2026/seed_42/moe_final.npz")
rows, optimizer, rng = train_phase(model, get_data(meta["split_seed"]),
    epochs=10, seed=0, phase="resumed", global_offset=meta["epoch"],
    lr=optimizer.lr, optimizer=optimizer, rng=rng)
```

The stored optimizer and RNG preserve the next update exactly on the same
numerical runtime. A new learning rate can be chosen for further exploration,
but should be reported separately from this fixed experimental protocol.

## Limitations and references

This is a small affine digit classifier, not a transformer or language model.
There are three overlapping splits of one small dataset; neither robustness
on a new dataset nor large-model behavior is established. The projection
budget omits several operations, top-k derivatives are local to fixed sets,
and the baseline hyperparameters have not been exhaustively optimized.
A capacity-matched dense nonlinear model and another dataset would strengthen
future research and are potential extensions to this experiment.

- [scikit-learn digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- [Komatsuzaki et al., Sparse Upcycling](https://arxiv.org/abs/2212.05055)
- [Fedus et al., Switch Transformers](https://arxiv.org/abs/2101.03961)
- [nbclient execution documentation](https://nbclient.readthedocs.io/en/latest/client.html)

The papers motivate weight reuse and balancing. This is an independent
educational experiment, not a reproduction of their architectures or results.
AI assistance was used to develop and document the implementation. Reported
numbers come from executed runs, saved checkpoints and numerical checks.
"""
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/EXPERIMENT.md").write_text(report)
    readme = fr"""<p align="center"><img src="assets/hero.svg" alt="Linear Becomes Experts — train, convert, keep learning" width="100%"></p>

# Linear Becomes Experts

**Train a literal linear model. Upcycle its learned weights into a sparse MoE. Measure continued learning.**

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/linear_to_moe.ipynb)
[![Verify submission](https://github.com/{REPO}/actions/workflows/verify.yml/badge.svg)](https://github.com/{REPO}/actions/workflows/verify.yml)

[Executed notebook](linear_to_moe.ipynb) · [Full report](docs/EXPERIMENT.md) · [Reproducibility guide](docs/REPRODUCIBILITY.md) · [Raw comparisons](results/comparison.csv)

## The measured result

A **650-parameter affine classifier** learns handwritten digits. Its trained
weights become **four independent experts**, with **two selected per image**.
Conversion preserves predictions to floating-point precision. Continued
training reduces both train and validation loss in **all {n} cases** across
**{len(cfg['seeds'])} initialization seeds and {len(cfg['split_seeds'])} stratified splits**.

{stage_table}
Mean ± sample SD; cross-entropy in nats. After conversion, mean training loss
falls **{drop_train:.2f}%** and validation loss falls **{drop_val:.2f}%**. Final mean MoE test
accuracy is **{100*a['moe_final_test']['accuracy']['mean']:.2f}%**. This is a small digit-classification
experiment with descriptive variation across overlapping splits.

![Measured learning curves](results/loss_curves.png)

## Baseline comparisons

{baseline_table}
Every run includes equal-step linear continuation, a linear control with the
same cumulative **forward projection MAC budget**, an independently initialized
MoE trained from scratch, and an upcycled MoE without balancing.
The projection budget includes pretraining but excludes backward, optimizer,
sorting, softmax and mixing; it is not a full FLOP or wall-clock match.

The upcycled MoE has lower validation CE than the equal-step linear control
in **{a['paired_comparisons']['linear']['val_ce']['moe_better']}/{n}** cases. Accuracy gains vary by seed and split;
balancing does not consistently improve held-out results. The [full report](docs/EXPERIMENT.md)
includes every paired delta, split mean, budget and unfavorable ablation result.

![Baseline comparisons](results/baseline_comparison.png)

## Run and verify

Use **Python 3.12** from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python check_submission.py --reproduce --check-singleton --execute-notebook
```

Windows activation: `.venv\Scripts\activate`.
For a separate experiment, preserving the submitted evidence:

```bash
python train.py --out runs/my-experiment
python -c "from plots import plot_results; plot_results('runs/my-experiment')"
```

For a quick single-seed/single-split run:

```bash
python train.py --out runs/quick --seeds 42 --split-seeds 2026
```

The standalone notebook embeds all code. Open in Colab/Jupyter and **Run all**.
Saved notebook outputs were executed by **a fresh Jupyter kernel**, not an
emulated executor. It also reloads a final checkpoint and trains ten more epochs.
No GPU, API key or dataset download is needed once dependencies are installed.

Verification independently checks checkpoint metrics, exact data splits, raw
CSV endpoints, aggregates, conversion, sparse dispatch, finite-difference
gradients, expert participation and exact model/Adam/RNG resumption. Fresh runs
use explicit floating-point tolerances (`rtol=1e-6`, `atol=1e-8`) and exclude
runtime metrics. Singleton plots are checked for numerical warnings.
GitHub Actions executes this verification on pushes and pull requests.

## Why conversion preserves predictions

The starting model has **no hidden layer**: $z(x)=xW+b$. Each expert receives
an independent copy of $W,b$, and normalized top-2 gates mix its logits:

$$z_{{MoE}}(x)=\sum_{{e\in S(x)}}g_e(x)(xW_e+b_e),\quad \sum_{{e\in S(x)}}g_e(x)=1.$$

Initially all expert logits agree, so the mixture equals the trained linear
model. Maximum measured conversion error: **{max_delta:.3e}**.
Different assignments produce different expert updates. The task router
gradient starts at zero, then appears as experts diverge. The balancing term
can train the router from the start. Hard selection itself is not differentiated.

| Model | Total parameters | Parameter entries used per image | Projection MACs per image |
|---|---:|---:|---:|
| Linear | 650 | 650 | 640 |
| Four-expert top-2 MoE | 2,860 | 1,560 | 1,536 |

Only selected experts process an image; no inputs are dropped. Parameter
counts are capacity counts and do not establish inference speed.

## Training protocol

{settings}
All continuation branches reset Adam. Identical seeds generate corresponding
minibatch schedules. Scaling uses a known pixel bound and no fitted transform.
Final epochs are fixed; no best-seed, best-checkpoint or test-driven tuning.

## Inspect the experts

All experts receive assignments and change weights. The router learns too.
The report includes validation class×expert heatmaps, normalized routing
entropy, gradient probes and the balancing ablation. Expert identities remain
local to each run, and routing patterns are not treated as semantic labels.

![Router diagnostics](results/router_diagnostics.png)

## Repository guide

| File | Purpose |
|---|---|
| [linear_to_moe.ipynb](linear_to_moe.ipynb) | Standalone notebook with real Jupyter outputs and five plots |
| [train.py](train.py) | Affine models, sparse routing, gradients, Adam, five-branch experiments |
| [verify.py](verify.py) | Numerical gradients, dispatch, independent storage, exact resumption |
| [check_submission.py](check_submission.py) | Saved-evidence audit, script reproduction and clean notebook execution |
| [evidence.py](evidence.py) | Float-tolerant comparisons with strict structure and identifiers |
| [plots.py](plots.py) | Dynamic plots; one-seed runs supported |
| [build_artifacts.py](build_artifacts.py) | Fresh-kernel notebook execution and measured documentation |
| [results/](results/) | Logs, per-epoch CSVs, all checkpoints, split indices and routing diagnostics |
| [docs/EXPERIMENT.md](docs/EXPERIMENT.md) | Method, all cases, paired comparisons, budgets and limitations |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Reproduction commands and saved-result navigation |
| [.github/workflows/verify.yml](.github/workflows/verify.yml) | Automated full verification |
| [MANIFEST.sha256](MANIFEST.sha256) | Submission file checksums |

## Scope and attribution

This demonstrates affine-model upcycling and continued learning on one small
real dataset. It does not establish language-model scaling, inference speedup
or universal MoE superiority. More datasets and a capacity-matched nonlinear
dense control would broaden the experimental evidence.

References: [digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html),
[Sparse Upcycling](https://arxiv.org/abs/2212.05055),
[Switch Transformers](https://arxiv.org/abs/2101.03961).
This is an independent educational experiment. AI assistance was used to
develop and document it; results come from executed training and verification.
"""
    (ROOT / "README.md").write_text(readme)
    submission = f"""# Submission

**Repository:** [github.com/{REPO}](https://github.com/{REPO})

**Notebook:** [Open in Colab](https://colab.research.google.com/github/{REPO}/blob/main/linear_to_moe.ipynb)

## Summary to submit

I trained a literal 650-parameter linear classifier on handwritten digits and
converted its learned weights into four independent experts with top-2 routing.
Conversion preserves the logits to floating-point precision. Across five
initialization seeds and three stratified splits ({n} paired cases), continued
MoE training reduces mean train CE from {a['moe_initial_train_ce']['mean']:.6f} to
{a['moe_final_train_ce']['mean']:.6f} and mean validation CE from
{a['moe_initial_val_ce']['mean']:.6f} to {a['moe_final_val_ce']['mean']:.6f}.
Both losses decrease in every case. Final mean test accuracy is
{100*a['moe_final_test']['accuracy']['mean']:.2f}%.

The repository includes a notebook executed in a fresh Jupyter kernel,
all raw logs and resumable checkpoints, equal-step and projection-budget
linear controls, a scratch MoE, a balancing ablation, class routing/entropy
and gradient diagnostics, and automated verification. Accuracy advantages
vary, and all baselines and unfavorable outcomes are retained.

## Project entry points

1. [Reproducibility guide](docs/REPRODUCIBILITY.md)
2. [Executed notebook](linear_to_moe.ipynb)
3. [Full report](docs/EXPERIMENT.md)
4. [Raw comparison CSV](results/comparison.csv)

The archive contains the repository contents in `linear-to-moe/`. The GitHub
repository uses `main`; Colab and CI links point to that exact repository.
"""
    (ROOT / "SUBMISSION.md").write_text(submission)
    requirements = ["numpy", "scikit-learn", "matplotlib", "threadpoolctl"]
    (ROOT / "requirements.txt").write_text("\n".join(f"{name}=={importlib.metadata.version(name)}" for name in requirements) + "\n")
    (ROOT / "requirements-dev.txt").write_text("-r requirements.txt\n" + "\n".join(
        f"{name}=={importlib.metadata.version(name)}" for name in ("nbformat", "nbclient", "ipykernel")) + "\n")
    # Re-derive the convenience CSV from already-measured summaries.
    rows = [{"split_seed": s["split_seed"], "seed": s["seed"], "model": name,
        "epochs": b["epochs"], "train_ce": b["train"]["ce"], "val_ce": b["val"]["ce"],
        "test_ce": b["test"]["ce"], "test_accuracy": b["test"]["accuracy"],
        "forward_projection_macs": b["forward_projection_macs"]}
        for s in summaries for name, b in s["branches"].items()]
    with (ROOT / "results/comparison.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Prepare an unexecuted notebook for CI bootstrap.")
    args = parser.parse_args()
    build_notebook(execute=not args.prepare_only)
    write_report()
    if args.prepare_only:
        path = ROOT / "README.md"
        text = path.read_text().replace("[Executed notebook]", "[Notebook source]")
        text = text.replace("Saved notebook outputs were executed by **a fresh Jupyter kernel**, not an\nemulated executor. It also reloads a final checkpoint and trains ten more epochs.",
            "Fresh Jupyter notebook execution is pending in CI. Its cells include a\ncheckpoint reload and ten-epoch continuation check.")
        path.write_text("> **Notebook validation pending:** the initial CI run will generate and check its Jupyter outputs. Script results below are already measured.\n\n" + text)
    print("Notebook and measured documentation saved.")
