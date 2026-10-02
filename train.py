"""Train a literal linear classifier, upcycle to top-2 MoE, continue both.

NumPy implements the model, exact gradients, and Adam. scikit-learn is used
only for its bundled digits dataset and stratified splitting, never training.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=-1, keepdims=True)


def cross_entropy(logits, labels):
    shifted = logits - logits.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(shifted).sum(axis=1))
                         - shifted[np.arange(len(labels)), labels]))


class Linear:
    """Affine 64 -> 10 classifier; no hidden layer or activation."""
    def __init__(self, seed=42, d=64, classes=10):
        rng = np.random.default_rng(seed)
        self.p = {"W": rng.normal(0, 0.01, (d, classes)),
                  "b": np.zeros(classes)}

    def forward(self, x):
        return x @ self.p["W"] + self.p["b"], None

    def loss_grad(self, x, y, alpha=0.0):
        logits, _ = self.forward(x)
        dz = softmax(logits)
        dz[np.arange(len(y)), y] -= 1
        dz /= len(y)
        ce = cross_entropy(logits, y)
        return ce, {"W": x.T @ dz, "b": dz.sum(axis=0)}, {"ce": ce, "balance": 0.0}


class MoE:
    """Independent copied linear experts with genuine sparse top-k dispatch.

    Only the selected experts process a sample. No capacity limit or dropping.
    Mix logits using softmax over selected router scores, then classify.
    """
    def __init__(self, dense, seed=1042, experts=4, top_k=2):
        if not 1 < top_k <= experts:
            raise ValueError("Use at least two selected experts for task router gradients.")
        d, c = dense.p["W"].shape
        self.experts, self.top_k = experts, top_k
        rng = np.random.default_rng(seed)
        self.p = {"W": np.repeat(dense.p["W"][None], experts, axis=0),
                  "b": np.repeat(dense.p["b"][None], experts, axis=0),
                  "R": rng.normal(0, 0.05, (d, experts)),
                  "r": np.zeros(experts)}

    @classmethod
    def from_scratch(cls, seed, d=64, classes=10, experts=4, top_k=2):
        """Same architecture, with independently initialized untrained experts."""
        model = cls(Linear(seed, d, classes), seed + 1000, experts, top_k)
        rng = np.random.default_rng(seed + 3000)
        model.p["W"] = rng.normal(0, 0.01, (experts, d, classes))
        return model

    def forward(self, x):
        scores = x @ self.p["R"] + self.p["r"]
        selected = np.argsort(-scores, axis=1, kind="stable")[:, :self.top_k]
        weights = softmax(np.take_along_axis(scores, selected, axis=1))
        expert_logits = np.zeros((len(x), self.top_k, self.p["b"].shape[1]))
        for e in range(self.experts):
            rows, slots = np.where(selected == e)
            expert_logits[rows, slots] = x[rows] @ self.p["W"][e] + self.p["b"][e]
        logits = (weights[:, :, None] * expert_logits).sum(axis=1)
        cache = (selected, weights, expert_logits, softmax(scores))
        return logits, cache

    def loss_grad(self, x, y, alpha=0.01):
        logits, (selected, weights, expert_logits, full_probs) = self.forward(x)
        n = len(y)
        dz = softmax(logits)
        dz[np.arange(n), y] -= 1
        dz /= n
        grads = {name: np.zeros_like(value) for name, value in self.p.items()}
        for e in range(self.experts):
            rows, slots = np.where(selected == e)
            de = dz[rows] * weights[rows, slots, None]
            grads["W"][e] = x[rows].T @ de
            grads["b"][e] = de.sum(axis=0)
        # Within a fixed top-k set, differentiate its normalized softmax weights.
        dweights = (expert_logits * dz[:, None, :]).sum(axis=2)
        dselected_scores = weights * (dweights - (weights * dweights).sum(axis=1, keepdims=True))
        dscores = np.zeros_like(full_probs)
        np.add.at(dscores, (np.arange(n)[:, None], selected), dselected_scores)
        # Switch-style auxiliary balancing; hard assignments are stop-gradient.
        f = np.bincount(selected.ravel(), minlength=self.experts) / (n * self.top_k)
        balance = float(self.experts * np.dot(f, full_probs.mean(axis=0)))
        dscores += alpha * self.experts / n * full_probs * (
            f[None, :] - (full_probs * f).sum(axis=1, keepdims=True))
        grads["R"], grads["r"] = x.T @ dscores, dscores.sum(axis=0)
        ce = cross_entropy(logits, y)
        return ce + alpha * balance, grads, {"ce": ce, "balance": balance}


class Adam:
    def __init__(self, params, lr=0.01, router_lr=0.002):
        self.lr, self.router_lr, self.t = lr, router_lr, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for name, grad in grads.items():
            if not np.all(np.isfinite(grad)):
                raise FloatingPointError(f"Nonfinite gradient: {name}")
            self.m[name] = 0.9 * self.m[name] + 0.1 * grad
            self.v[name] = 0.999 * self.v[name] + 0.001 * grad ** 2
            mhat = self.m[name] / (1 - 0.9 ** self.t)
            vhat = self.v[name] / (1 - 0.999 ** self.t)
            lr = self.router_lr if name in ("R", "r") else self.lr
            params[name] -= lr * mhat / (np.sqrt(vhat) + 1e-8)


def get_data(split_seed=2026):
    x, y = load_digits(return_X_y=True)
    x = x.astype(np.float64) / 16.0
    indices = np.arange(len(y))
    train, rest = train_test_split(indices, test_size=0.4, random_state=split_seed, stratify=y)
    val, test = train_test_split(rest, test_size=0.5, random_state=split_seed, stratify=y[rest])
    assert not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test))
    return {"train": (x[train], y[train]), "val": (x[val], y[val]),
            "test": (x[test], y[test]), "indices": {"train": train, "val": val, "test": test},
            "sha256": hashlib.sha256(x.tobytes() + y.astype(np.int64).tobytes()).hexdigest()}


def evaluate(model, data):
    x, y = data
    z, _ = model.forward(x)
    return {"ce": cross_entropy(z, y), "accuracy": float(np.mean(z.argmax(axis=1) == y))}


def save_checkpoint(path, model, optimizer=None, rng=None, metadata=None):
    meta = dict(metadata or {})
    meta.update({"model": type(model).__name__, "experts": getattr(model, "experts", 1),
                 "top_k": getattr(model, "top_k", 1)})
    arrays = {"param_" + k: v for k, v in model.p.items()}
    if optimizer is not None:
        meta["adam"] = {"t": optimizer.t, "lr": optimizer.lr, "router_lr": optimizer.router_lr}
        arrays.update({"adam_m_" + k: v for k, v in optimizer.m.items()})
        arrays.update({"adam_v_" + k: v for k, v in optimizer.v.items()})
    if rng is not None:
        meta["rng"] = rng.bit_generator.state
    arrays["metadata_json"] = np.array(json.dumps(meta))
    np.savez_compressed(path, **arrays)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata_json"]))
        model = Linear(d=z["param_W"].shape[-2], classes=z["param_W"].shape[-1])
        if meta["model"] == "MoE":
            model = MoE(model, experts=meta["experts"], top_k=meta["top_k"])
        model.p = {k: z["param_" + k].copy() for k in model.p}
        optimizer = None
        if "adam" in meta:
            cfg = meta["adam"]
            optimizer = Adam(model.p, cfg["lr"], cfg["router_lr"])
            optimizer.t = cfg["t"]
            optimizer.m = {k: z["adam_m_" + k].copy() for k in model.p}
            optimizer.v = {k: z["adam_v_" + k].copy() for k in model.p}
    rng = np.random.default_rng()
    if "rng" in meta:
        rng.bit_generator.state = meta["rng"]
    return model, optimizer, rng, meta


def train_phase(model, data, epochs, seed, phase, global_offset, lr, batch_size=64,
                alpha=0.01, router_lr=0.002, verbose=True, optimizer=None, rng=None):
    optimizer = optimizer or Adam(model.p, lr, router_lr)
    rng = rng or np.random.default_rng(seed)
    rows = []
    start = time.perf_counter()
    training_seconds = 0.0
    x, y = data["train"]
    for epoch in range(epochs + 1):
        if epoch:
            training_start = time.perf_counter()
            order = rng.permutation(len(y))
            for ix in range(0, len(y), batch_size):
                batch = order[ix:ix + batch_size]
                _, gradients, _ = model.loss_grad(x[batch], y[batch], alpha)
                optimizer.step(model.p, gradients)
            training_seconds += time.perf_counter() - training_start
        train, val = evaluate(model, data["train"]), evaluate(model, data["val"])
        row = {"phase": phase, "epoch": epoch, "global_epoch": epoch + global_offset,
               "train_ce": train["ce"], "val_ce": val["ce"],
               "train_accuracy": train["accuracy"], "val_accuracy": val["accuracy"],
               "elapsed_seconds": time.perf_counter() - start,
               "training_seconds": training_seconds}
        if isinstance(model, MoE):
            _, cache = model.forward(x)
            selected, weights, _, probs = cache
            f = np.bincount(selected.ravel(), minlength=model.experts) / selected.size
            row["balance_loss"] = float(model.experts * np.dot(f, probs.mean(axis=0)))
            row["objective"] = train["ce"] + alpha * row["balance_loss"]
            row["router_entropy_normalized"] = float(np.mean(
                -np.sum(probs * np.log(np.maximum(probs, 1e-300)), axis=1)) / np.log(model.experts))
            row["selected_gate_entropy_normalized"] = float(np.mean(
                -np.sum(weights * np.log(np.maximum(weights, 1e-300)), axis=1)) / np.log(model.top_k))
            if epoch == 0 or epoch == epochs or epoch % 20 == 0:
                task_grad = model.loss_grad(x, y, 0.0)[1]
                total_grad = model.loss_grad(x, y, alpha)[1]
                row["task_router_grad_l2"] = float(np.linalg.norm(task_grad["R"]))
                row["aux_router_grad_l2"] = float(np.linalg.norm(total_grad["R"] - task_grad["R"]))
            for e in range(model.experts):
                row[f"expert_{e}_assignment_share"] = float(f[e])
        rows.append(row)
        if verbose and (epoch == 0 or epoch == epochs or epoch % 20 == 0):
            print(f"{phase:18s} epoch={epoch:3d} train_CE={train['ce']:.6f} val_CE={val['ce']:.6f}", flush=True)
    return rows, optimizer, rng



def routing_diagnostics(model, data):
    """Validation-only class routing; columns are case-local expert identities."""
    x, y = data["val"]
    _, (selected, weights, _, probs) = model.forward(x)
    class_counts = np.array([np.bincount(selected[y == c].ravel(),
        minlength=model.experts) for c in range(model.p["b"].shape[1])])
    return {"split": "validation", "class_labels": list(range(len(class_counts))),
        "class_counts": class_counts.tolist(),
        "class_assignment_shares": (class_counts / class_counts.sum(axis=1, keepdims=True)).tolist(),
        "class_sample_counts": np.bincount(y, minlength=len(class_counts)).tolist(),
        "router_entropy_normalized": float(np.mean(-np.sum(probs * np.log(np.maximum(probs, 1e-300)), axis=1)) / np.log(model.experts)),
        "selected_gate_entropy_normalized": float(np.mean(-np.sum(weights * np.log(np.maximum(weights, 1e-300)), axis=1)) / np.log(model.top_k))}


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "n": len(values)}


def aggregate_summaries(summaries):
    """Descriptive SD, never an iid claim about overlapping data splits."""
    numeric = [k for k, v in summaries[0].items()
               if isinstance(v, (int, float)) and k not in ("seed", "split_seed")]
    result = {k: stats([s[k] for s in summaries]) for k in numeric}
    for stage in ("linear_pretrain_test", "linear_final_test", "moe_final_test"):
        result[stage] = {m: stats([s[stage][m] for s in summaries]) for m in ("ce", "accuracy")}
    result["branches"] = {name: {split: {m: stats([s["branches"][name][split][m] for s in summaries])
        for m in ("ce", "accuracy")} for split in ("train", "val", "test")}
        for name in summaries[0]["branches"]}
    comparisons = {}
    for baseline in ("linear", "linear_mac_matched", "scratch_moe", "moe_no_balance"):
        comparisons[baseline] = {}
        for split, metric in (("val", "ce"), ("test", "ce"), ("test", "accuracy")):
            values = [s["branches"]["moe"][split][metric] - s["branches"][baseline][split][metric]
                      for s in summaries]
            name = f"{split}_{metric}"
            comparisons[baseline][name] = {**stats(values),
                "moe_better": sum(v < -1e-12 if metric == "ce" else v > 1e-12 for v in values),
                "ties": sum(abs(v) <= 1e-12 for v in values),
                "per_split_mean_delta": {str(seed): float(np.mean([v for s, v in zip(summaries, values)
                    if s["split_seed"] == seed])) for seed in sorted({s["split_seed"] for s in summaries})}}
    result["paired_comparisons"] = comparisons
    result["per_split"] = {str(seed): {name: {split: stats([s["branches"][name][split]["ce"]
        for s in summaries if s["split_seed"] == seed]) for split in ("train", "val", "test")}
        for name in summaries[0]["branches"]} for seed in sorted({s["split_seed"] for s in summaries})}
    return result


def run_seed(seed, data, out, pre_epochs=20, post_epochs=80, verbose=True,
             split_seed=2026, experts=4, top_k=2, batch_size=64, alpha=0.01):
    seed_dir = out / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    common = {"batch_size": batch_size, "router_lr": 0.002, "verbose": verbose}
    linear = Linear(seed)
    pre, opt, rng = train_phase(linear, data, pre_epochs, seed + 100,
                               "linear_pretrain", 0, 0.01, **common)
    meta = {"seed": seed, "split_seed": split_seed, "epoch": pre_epochs}
    save_checkpoint(seed_dir / "linear_pretrained.npz", linear, opt, rng, meta)
    converted = MoE(linear, seed + 1000, experts, top_k)
    all_x = np.concatenate([data[s][0] for s in ("train", "val", "test")])
    delta = float(np.max(np.abs(linear.forward(all_x)[0] - converted.forward(all_x)[0])))
    if delta > 1e-12:
        raise AssertionError(f"Conversion changed logits: {delta}")
    save_checkpoint(seed_dir / "moe_converted.npz", converted, metadata=meta)
    d, c = linear.p["W"].shape
    linear_macs, moe_macs = d * c, d * experts + top_k * d * c
    # Matches affine projection MACs; excludes gates, mixing, sorting, backward,
    # and optimizer work. The rounded-up dense budget is explicitly recorded.
    mac_epochs = int(np.ceil(post_epochs * moe_macs / linear_macs))
    models = {"linear": copy.deepcopy(linear), "moe": copy.deepcopy(converted),
              "linear_mac_matched": copy.deepcopy(linear),
              "moe_no_balance": copy.deepcopy(converted),
              "scratch_moe": MoE.from_scratch(seed, d, c, experts, top_k)}
    scratch_pre, _, _ = train_phase(models["scratch_moe"], data, pre_epochs,
        seed + 100, "scratch_pretrain", 0, 0.01, alpha=alpha, **common)
    phase_names = {"linear": "linear_continue", "moe": "moe_continue",
        "linear_mac_matched": "linear_mac_matched", "moe_no_balance": "moe_no_balance",
        "scratch_moe": "scratch_continue"}
    histories, phase_rows, branches = pre + scratch_pre, {}, {}
    for name, model in models.items():
        epochs = mac_epochs if name == "linear_mac_matched" else post_epochs
        branch_alpha = 0.0 if name == "moe_no_balance" else alpha
        rows, optimizer, batch_rng = train_phase(model, data, epochs, seed + 200,
            phase_names[name], pre_epochs, 0.005, alpha=branch_alpha, **common)
        histories += rows
        phase_rows[name] = rows
        save_checkpoint(seed_dir / f"{name}_final.npz", model, optimizer, batch_rng,
            {"seed": seed, "split_seed": split_seed, "epoch": pre_epochs + epochs,
             "continuation_epochs": epochs, "balance_alpha": branch_alpha})
        pre_macs = moe_macs if name == "scratch_moe" else linear_macs
        post_macs = moe_macs if isinstance(model, MoE) else linear_macs
        pre_seconds = scratch_pre[-1]["training_seconds"] if name == "scratch_moe" else pre[-1]["training_seconds"]
        branches[name] = {"train": evaluate(model, data["train"]),
            "val": evaluate(model, data["val"]), "test": evaluate(model, data["test"]),
            "epochs": pre_epochs + epochs, "continuation_epochs": epochs,
            "forward_projection_macs": len(data["train"][1]) * (pre_epochs * pre_macs + epochs * post_macs),
            "training_seconds": pre_seconds + rows[-1]["training_seconds"]}
    for row in histories:
        row.update({"seed": seed, "split_seed": split_seed})
    columns = list(dict.fromkeys(k for row in histories for k in row))
    with (seed_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(histories)
    moe = models["moe"]
    summary = {"seed": seed, "split_seed": split_seed,
        "linear_params": sum(v.size for v in linear.p.values()),
        "moe_total_params": sum(v.size for v in moe.p.values()),
        "moe_active_params_per_sample": top_k * (moe.p["W"][0].size + moe.p["b"][0].size) + moe.p["R"].size + moe.p["r"].size,
        "linear_projection_macs_per_sample": linear_macs,
        "moe_projection_macs_per_sample": moe_macs,
        "conversion_max_abs_logit_difference": delta,
        "initial_train_ce": pre[0]["train_ce"], "initial_val_ce": pre[0]["val_ce"],
        "linear_pretrain_train_ce": pre[-1]["train_ce"], "linear_pretrain_val_ce": pre[-1]["val_ce"],
        "linear_final_train_ce": branches["linear"]["train"]["ce"],
        "linear_final_val_ce": branches["linear"]["val"]["ce"],
        "moe_initial_train_ce": phase_rows["moe"][0]["train_ce"],
        "moe_initial_val_ce": phase_rows["moe"][0]["val_ce"],
        "moe_final_train_ce": branches["moe"]["train"]["ce"],
        "moe_final_val_ce": branches["moe"]["val"]["ce"],
        "linear_pretrain_test": evaluate(linear, data["test"]),
        "linear_final_test": branches["linear"]["test"], "moe_final_test": branches["moe"]["test"],
        "router_weight_change_l2": float(np.linalg.norm(moe.p["R"] - converted.p["R"])),
        "expert_weight_change_l2": [float(np.linalg.norm(moe.p["W"][e] - converted.p["W"][e])) for e in range(experts)],
        "mean_pairwise_expert_weight_distance": float(np.mean([
            np.linalg.norm(moe.p["W"][a] - moe.p["W"][b]) for a in range(experts) for b in range(a + 1, experts)])),
        "final_training_assignment_share": [phase_rows["moe"][-1][f"expert_{e}_assignment_share"] for e in range(experts)],
        "branches": branches}
    for split in ("train", "val"):
        assert summary[f"linear_pretrain_{split}_ce"] < summary[f"initial_{split}_ce"]
        assert summary[f"linear_final_{split}_ce"] < summary[f"linear_pretrain_{split}_ce"]
        assert summary[f"moe_final_{split}_ce"] < summary[f"moe_initial_{split}_ce"]
    assert summary["router_weight_change_l2"] > 0
    assert min(summary["expert_weight_change_l2"]) > 0
    diagnostics = {name: routing_diagnostics(models[name], data)
        for name in ("moe", "scratch_moe", "moe_no_balance")}
    (seed_dir / "routing.json").write_text(json.dumps(diagnostics, indent=2) + "\n")
    (seed_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, histories


def run_experiment(out="results", seeds=(42, 43, 44, 45, 46), pre_epochs=20,
                   post_epochs=80, verbose=True, split_seeds=(2026, 2027, 2028),
                   experts=4, top_k=2, batch_size=64, alpha=0.01):
    if not seeds or not split_seeds or len(set(seeds)) != len(seeds) or len(set(split_seeds)) != len(split_seeds):
        raise ValueError("Provide nonempty, unique initialization seeds and split seeds.")
    if min(pre_epochs, post_epochs, batch_size) < 1 or alpha < 0 or not 1 < top_k <= experts:
        raise ValueError("Positive epochs/batch size, nonnegative alpha, and 1 < top_k <= experts required.")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = get_data(split_seeds[0])
    config = {"schema_version": 2, "seeds": list(seeds), "split_seeds": list(split_seeds),
        "pretrain_epochs": pre_epochs, "continuation_epochs": post_epochs, "batch_size": batch_size,
        "pretrain_lr": 0.01, "continuation_lr": 0.005, "router_lr": 0.002,
        "balance_alpha": alpha, "experts": experts, "top_k": top_k,
        "dtype": "float64", "device": "CPU", "blas_threads": 1,
        "split_counts": {s: len(data[s][1]) for s in ("train", "val", "test")},
        "dataset_sha256": data["sha256"], "python": platform.python_version(),
        "numpy": np.__version__, "scikit_learn": sklearn.__version__,
        "hardware": platform.platform(), "cpu_count": os.cpu_count(),
        "note": "Fixed final epochs; no checkpoint selection or test tuning. Adam resets after pretraining in every branch. Projection MAC budget excludes backward, optimizer, gating, sorting and mixing. Splits overlap; variation is descriptive."}
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    summaries, histories = [], []
    with threadpool_limits(limits=1):
        for split_seed in split_seeds:
            data = get_data(split_seed)
            split_dir = out / f"split_{split_seed}"
            split_dir.mkdir(exist_ok=True)
            np.savez_compressed(split_dir / "split_indices.npz", **data["indices"])
            for seed in seeds:
                if verbose:
                    print(f"\n=== Split {split_seed} / seed {seed} ===", flush=True)
                summary, history = run_seed(seed, data, split_dir, pre_epochs, post_epochs,
                    verbose, split_seed, experts, top_k, batch_size, alpha)
                summaries.append(summary)
                histories.extend(history)
    (out / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (out / "aggregate.json").write_text(json.dumps(aggregate_summaries(summaries), indent=2) + "\n")
    comparison = [{"split_seed": s["split_seed"], "seed": s["seed"], "model": name,
        "epochs": b["epochs"], "train_ce": b["train"]["ce"], "val_ce": b["val"]["ce"],
        "test_ce": b["test"]["ce"], "test_accuracy": b["test"]["accuracy"],
        "forward_projection_macs": b["forward_projection_macs"]}
        for s in summaries for name, b in s["branches"].items()]
    with (out / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    print("\nAll runs passed loss-decrease and conversion assertions.", flush=True)
    return summaries, histories


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[2026, 2027, 2028])
    parser.add_argument("--pre-epochs", type=int, default=20)
    parser.add_argument("--post-epochs", type=int, default=80)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--balance-alpha", type=float, default=0.01)
    args = parser.parse_args()
    run_experiment(args.out, args.seeds, args.pre_epochs, args.post_epochs,
        split_seeds=args.split_seeds, experts=args.experts, top_k=args.top_k,
        batch_size=args.batch_size, alpha=args.balance_alpha)
