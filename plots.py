"""Plot measured CSV/JSON evidence. No smoothing; singleton SD is zero."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {"pre": "#5968d8", "linear": "#bb7633", "moe": "#008b82",
          "linear_mac_matched": "#ce995c", "scratch_moe": "#775bc4", "moe_no_balance": "#bc5273"}
LABELS = {"linear": "Linear · 100 epochs", "moe": "Upcycled MoE · 100",
          "linear_mac_matched": "Linear · projection budget", "scratch_moe": "Scratch MoE · 100",
          "moe_no_balance": "Upcycled MoE · no balance"}


def sample_sd(values, axis=0):
    a = np.asarray(values)
    return a.std(axis=axis, ddof=1) if a.shape[axis] > 1 else np.zeros_like(a.mean(axis=axis))


def plot_results(folder="results"):
    folder = Path(folder)
    summaries = json.loads((folder / "summary.json").read_text())
    cfg = json.loads((folder / "config.json").read_text())
    histories = []
    for s in summaries:
        with (folder / f"split_{s['split_seed']}" / f"seed_{s['seed']}" / "history.csv").open() as f:
            histories.append(list(csv.DictReader(f)))
    n, e = len(summaries), cfg["experts"]
    caption = f"{len(cfg['seeds'])} initialization seed(s) × {len(cfg['split_seeds'])} split(s) = {n} paired run(s)"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
        "text.color": "#192936", "axes.edgecolor": "#bac5ce", "savefig.facecolor": "white"})
    paths = []

    def save(fig, name):
        fig.savefig(folder / (name + ".png"), dpi=170)
        fig.savefig(folder / (name + ".svg"))
        paths.append(folder / (name + ".png"))
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), layout="constrained")
    for ax, key, title in zip(axes, ("train_ce", "val_ce"), ("Training loss", "Validation loss")):
        for phase, color, label in (("linear_pretrain", COLORS["pre"], "Linear pretraining"),
                ("linear_continue", COLORS["linear"], "Linear continuation"),
                ("moe_continue", COLORS["moe"], "Upcycled MoE continuation")):
            curves = [[r for r in h if r["phase"] == phase] for h in histories]
            xs = [int(r["global_epoch"]) for r in curves[0]]
            ys = np.array([[float(r[key]) for r in rows] for rows in curves])
            mean, sd = ys.mean(axis=0), sample_sd(ys)
            ax.plot(xs, mean, color=color, linewidth=2.3, label=label)
            ax.fill_between(xs, np.maximum(mean - sd, 1e-8), mean + sd, color=color, alpha=.13)
        ax.axvline(cfg["pretrain_epochs"], color="#78838f", linestyle="--", linewidth=1)
        ax.text(cfg["pretrain_epochs"] + 2, .93, "Convert", transform=ax.get_xaxis_transform(), fontsize=9)
        ax.set(title=title, xlabel="Epoch within each branch", ylabel="Cross-entropy (nats)", yscale="log")
        ax.grid(axis="y", alpha=.15)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("A trained linear model becomes a trainable MoE", fontsize=18, weight="bold")
    fig.supxlabel(caption + " · mean ± sample SD; descriptive variation", fontsize=9)
    save(fig, "loss_curves")

    names = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.7), layout="constrained")
    for ax, split, metric, title in ((axes[0], "val", "ce", "Final validation cross-entropy"),
                                   (axes[1], "test", "accuracy", "Final test accuracy")):
        vals = np.array([[s["branches"][name][split][metric] for s in summaries] for name in names])
        scale = 100 if metric == "accuracy" else 1
        ax.barh(np.arange(len(names)), vals.mean(axis=1) * scale,
                xerr=sample_sd(vals, axis=1) * scale, color=[COLORS[name] for name in names],
                alpha=.9, capsize=3)
        ax.set(yticks=np.arange(len(names)), yticklabels=[LABELS[name] for name in names], title=title,
               xlabel="Accuracy (%)" if metric == "accuracy" else "Cross-entropy (nats)")
        ax.invert_yaxis()
        if metric == "accuracy":
            ax.set_xlim(max(0, 100 * vals.min() - 2), 100)
            ax.text(.02, .02, "Accuracy axis is truncated", transform=ax.transAxes, fontsize=8)
        ax.grid(axis="x", alpha=.15)
    fig.suptitle("Stronger baselines at fixed final checkpoints", fontsize=17, weight="bold")
    fig.supxlabel(caption + " · bars: mean ± sample SD · epoch counts differ for budget control", fontsize=9)
    # Epoch labels are derived from config, not assumed for custom runs.
    display_names = {"linear": "Linear", "moe": "Upcycled MoE",
        "linear_mac_matched": "Linear (projection budget)", "scratch_moe": "Scratch MoE",
        "moe_no_balance": "Upcycled MoE, no balance"}
    labels = [f"{display_names[name]} · {summaries[0]['branches'][name]['epochs']} epochs" for name in names]
    for ax in axes:
        ax.set_yticklabels(labels)
    save(fig, "baseline_comparison")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), layout="constrained")
    min_shares = [min(s["final_training_assignment_share"]) for s in summaries]
    axes[0].bar(np.arange(n), min_shares, color=COLORS["moe"], alpha=.85)
    axes[0].axhline(1 / e, color="#7c8690", ls="--", label="Uniform share")
    axes[0].set(xticks=np.arange(n), xticklabels=[f"{s['split_seed']}/{s['seed']}" for s in summaries],
        ylabel="Smallest expert assignment share", title="No expert is unused in the final run")
    axes[0].tick_params(axis="x", labelrotation=55, labelsize=8)
    axes[0].set_ylim(0, max(1 / e, max(min_shares)) * 1.25)
    axes[0].legend(frameon=False, fontsize=8)
    case = summaries[0]
    rows = [r for r in histories[0] if r["phase"] == "moe_continue"]
    for expert in range(e):
        axes[1].plot([int(r["epoch"]) for r in rows],
                     [float(r[f"expert_{expert}_assignment_share"]) for r in rows], label=f"Expert {expert}")
    axes[1].set(title=f"First configured case: {case['split_seed']}/{case['seed']}",
                xlabel="Epochs after conversion", ylabel="Share of selected assignments")
    axes[1].legend(ncol=2, frameon=False, fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("All experts participate in training", fontsize=18, weight="bold")
    fig.supxlabel("Expert identities are local to each run; curves show one predetermined case", fontsize=9)
    save(fig, "expert_routing")

    routing = json.loads((folder / f"split_{case['split_seed']}" / f"seed_{case['seed']}" / "routing.json").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9), layout="constrained")
    for ax, name, title in ((axes[0, 0], "moe", "Validation class routing: balanced MoE"),
                            (axes[0, 1], "moe_no_balance", "Validation class routing: no balance")):
        shares = np.array(routing[name]["class_assignment_shares"])
        im = ax.imshow(shares, vmin=0, vmax=1 / cfg["top_k"], cmap="YlGnBu", aspect="auto")
        ax.set(title=title, xlabel="Expert ID within this case", ylabel="Digit class",
               xticks=range(e), yticks=range(len(shares)))
        for cls in range(len(shares)):
            for expert in range(e):
                ax.text(expert, cls, f"{shares[cls, expert]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if shares[cls, expert] > .65 / cfg["top_k"] else "#18303a")
        fig.colorbar(im, ax=ax, shrink=.85, label="Share of class assignments")
    entropy_ax, grad_ax = axes[1]
    for phase, label, color in (("moe_continue", "Balanced", COLORS["moe"]),
            ("moe_no_balance", "No balance", COLORS["moe_no_balance"])):
        r = [row for row in histories[0] if row["phase"] == phase]
        xs = [int(row["epoch"]) for row in r]
        entropy_ax.plot(xs, [float(row["router_entropy_normalized"]) for row in r], color=color,
            label=label + " · all experts")
        entropy_ax.plot(xs, [float(row["selected_gate_entropy_normalized"]) for row in r],
            color=color, ls="--", label=label + " · selected gates")
    entropy_ax.set(title="Router uncertainty during training", xlabel="Epochs after conversion",
        ylabel="Entropy / log(number of choices)", ylim=(0, 1.03))
    entropy_ax.legend(frameon=False, fontsize=8)
    probes = [r for r in rows if r.get("task_router_grad_l2")]
    for key, label in (("task_router_grad_l2", "Task gradient"), ("aux_router_grad_l2", "Balance gradient")):
        grad_ax.plot([int(r["epoch"]) for r in probes], [float(r[key]) for r in probes], "o-", label=label)
    grad_ax.set(title="Router learns after experts diverge", xlabel="Epochs after conversion",
        ylabel="Full-training-set router gradient L2")
    grad_ax.legend(frameon=False, fontsize=8)
    for ax in axes[1]:
        ax.grid(axis="y", alpha=.15)
    fig.suptitle(f"Inspect the router · split {case['split_seed']} / seed {case['seed']}", fontsize=18, weight="bold")
    fig.supxlabel("Heatmaps use validation labels only; class differences are descriptive, not semantic expert labels", fontsize=9)
    save(fig, "router_diagnostics")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), layout="constrained")
    for ax, key, title in zip(axes, ("train_ce", "val_ce"), ("Training", "Validation")):
        for phase, name in (("linear_mac_matched", "linear_mac_matched"), ("moe_continue", "moe")):
            r = [[row for row in h if row["phase"] == phase] for h in histories]
            macs = case["linear_projection_macs_per_sample"] if name == "linear_mac_matched" else case["moe_projection_macs_per_sample"]
            budget = [cfg["pretrain_epochs"] * case["linear_projection_macs_per_sample"] + int(row["epoch"]) * macs for row in r[0]]
            values = np.array([[float(row[key]) for row in run] for run in r])
            ax.plot(np.array(budget) / 1000, values.mean(axis=0), color=COLORS[name], label=name.replace("_", " "))
            sd = sample_sd(values)
            ax.fill_between(np.array(budget) / 1000, np.maximum(values.mean(axis=0) - sd, 1e-8),
                values.mean(axis=0) + sd, color=COLORS[name], alpha=.12)
        ax.set(title=title + " loss vs projection budget", yscale="log", ylabel="Cross-entropy (nats)",
               xlabel="Cumulative affine projection MACs per training example (thousands)")
        ax.grid(axis="y", alpha=.15)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Compare the same forward projection budget", fontsize=17, weight="bold")
    fig.supxlabel("Includes linear pretraining · excludes backward, optimizer, sorting, softmax, and mixing costs", fontsize=9)
    save(fig, "compute_budget")
    return paths


if __name__ == "__main__":
    for path in plot_results():
        print(path)
