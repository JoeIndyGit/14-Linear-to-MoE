"""Verify saved evidence, optionally reproduce scripts or execute a fresh notebook.

Submitted results are read-only; fresh runs use isolated temporary directories.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import tempfile
import warnings

import numpy as np
from threadpoolctl import threadpool_limits
from train import (aggregate_summaries, evaluate, get_data, load_checkpoint,
                   routing_diagnostics, run_experiment)
from evidence import compare_evidence
from verify import verify

ROOT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text())


def close(actual, expected, label):
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8, err_msg=label)


def check_manifest():
    count = 0
    for line in (ROOT / "MANIFEST.sha256").read_text().splitlines():
        digest, name = line.split("  ", 1)
        file = (ROOT / name).resolve()
        assert file.is_relative_to(ROOT) and file.is_file(), name
        assert hashlib.sha256(file.read_bytes()).hexdigest() == digest, f"Changed file: {name}"
        count += 1
    return count


def check_local_links():
    count = 0
    for doc in [ROOT / "README.md", ROOT / "SUBMISSION.md", *sorted((ROOT / "docs").glob("*.md"))]:
        text = doc.read_text()
        links = re.findall(r"\]\(([^)]+)\)", text) + re.findall(r'<img[^>]+src="([^"]+)"', text)
        for link in links:
            if "://" in link or link.startswith("#"):
                continue
            target = (doc.parent / link.split("#", 1)[0]).resolve()
            assert target.exists(), f"Broken link in {doc.name}: {link}"
            count += 1
    return count


def check_notebook():
    notebook = read_json(ROOT / "linear_to_moe.ipynb")
    assert notebook["nbformat"] == 4
    code_cells = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    assert len(code_cells) >= 8
    figures, stdout = 0, ""
    for index, cell in enumerate(code_cells, 1):
        assert cell["execution_count"] == index
        compile("".join(cell["source"]), f"notebook_cell_{index}", "exec")
        assert "execution" in cell["metadata"], "Missing Jupyter execution timing"
        for output in cell["outputs"]:
            assert output["output_type"] != "error"
            if output["output_type"] == "stream":
                stdout += "".join(output["text"])
            if "image/png" in output.get("data", {}):
                figures += 1
    assert figures >= 5
    assert "All runs passed loss-decrease and conversion assertions." in stdout
    assert "All checks passed." in stdout
    for name in ("train.py", "evidence.py", "verify.py", "plots.py"):
        expected = (ROOT / name).read_text().split('if __name__ == "__main__":')[0]
        if name == "verify.py":
            expected = re.sub(r"^from (?:train|evidence) import .+\n", "", expected, flags=re.MULTILINE)
        if name == "plots.py":
            expected = expected.replace('matplotlib.use("Agg")\n', '')
        parts = ["".join(c["source"]) for c in code_cells if c["metadata"].get("module") == name]
        # Cells contain definitions only; calls are in separate cells.
        assert "".join(parts) == expected, f"Stale notebook source: {name}"
    return {"executed_cells": len(code_cells), "embedded_figures": figures}


def check_results():
    folder = ROOT / "results"
    cfg, summaries = read_json(folder / "config.json"), read_json(folder / "summary.json")
    expected_ids = [(split, seed) for split in cfg["split_seeds"] for seed in cfg["seeds"]]
    assert expected_ids == [(s["split_seed"], s["seed"]) for s in summaries]
    for split_seed in cfg["split_seeds"]:
        data = get_data(split_seed)
        assert cfg["dataset_sha256"] == data["sha256"]
        with np.load(folder / f"split_{split_seed}" / "split_indices.npz", allow_pickle=False) as indices:
            for split in ("train", "val", "test"):
                np.testing.assert_array_equal(indices[split], data["indices"][split])
                assert len(data[split][1]) == cfg["split_counts"][split]
    phase_names = {"linear": "linear_continue", "moe": "moe_continue",
        "linear_mac_matched": "linear_mac_matched", "scratch_moe": "scratch_continue",
        "moe_no_balance": "moe_no_balance"}
    for s in summaries:
        data = get_data(s["split_seed"])
        base = folder / f"split_{s['split_seed']}" / f"seed_{s['seed']}"
        compare_evidence(read_json(base / "summary.json"), s)
        pretrained, pre_opt, _, pre_meta = load_checkpoint(base / "linear_pretrained.npz")
        converted, _, _, _ = load_checkpoint(base / "moe_converted.npz")
        steps = int(np.ceil(len(data["train"][1]) / cfg["batch_size"]))
        assert pre_meta["epoch"] == cfg["pretrain_epochs"]
        assert pre_opt.t == cfg["pretrain_epochs"] * steps
        all_x = np.concatenate([data[k][0] for k in ("train", "val", "test")])
        close(converted.forward(all_x)[0], pretrained.forward(all_x)[0], "conversion logits")
        delta = float(np.max(np.abs(converted.forward(all_x)[0] - pretrained.forward(all_x)[0])))
        assert delta < 1e-12
        close(delta, s["conversion_max_abs_logit_difference"], "conversion error")
        for expert in range(converted.experts):
            np.testing.assert_array_equal(converted.p["W"][expert], pretrained.p["W"])
            np.testing.assert_array_equal(converted.p["b"][expert], pretrained.p["b"])
        for split in ("train", "val"):
            close(evaluate(pretrained, data[split])["ce"], s[f"linear_pretrain_{split}_ce"], "pretrained CE")
            assert s[f"linear_pretrain_{split}_ce"] < s[f"initial_{split}_ce"]
            assert s[f"linear_final_{split}_ce"] < s[f"linear_pretrain_{split}_ce"]
            assert s[f"moe_final_{split}_ce"] < s[f"moe_initial_{split}_ce"]
            close(s[f"moe_initial_{split}_ce"], s[f"linear_pretrain_{split}_ce"], "conversion CE")
        compare_evidence(evaluate(pretrained, data["test"]), s["linear_pretrain_test"])
        with (base / "history.csv").open() as f:
            history = list(csv.DictReader(f))
        for name, b in s["branches"].items():
            model, opt, _, meta = load_checkpoint(base / f"{name}_final.npz")
            assert opt.t == b["continuation_epochs"] * steps
            assert meta["epoch"] == b["epochs"] and meta["split_seed"] == s["split_seed"]
            for split in ("train", "val", "test"):
                compare_evidence(evaluate(model, data[split]), b[split])
            rows = [r for r in history if r["phase"] == phase_names[name]]
            assert [int(r["epoch"]) for r in rows] == list(range(b["continuation_epochs"] + 1))
            for split in ("train", "val"):
                close(float(rows[-1][f"{split}_ce"]), b[split]["ce"], "CSV endpoint")
                assert all(np.isfinite(float(r[f"{split}_ce"])) for r in rows)
            pre_macs = s["moe_projection_macs_per_sample"] if name == "scratch_moe" else s["linear_projection_macs_per_sample"]
            post_macs = s["moe_projection_macs_per_sample"] if name in ("moe", "scratch_moe", "moe_no_balance") else s["linear_projection_macs_per_sample"]
            assert b["forward_projection_macs"] == len(data["train"][1]) * (cfg["pretrain_epochs"] * pre_macs + b["continuation_epochs"] * post_macs)
            if name in ("moe", "scratch_moe", "moe_no_balance"):
                compare_evidence(routing_diagnostics(model, data), read_json(base / "routing.json")[name])
            if name == "moe":
                _, (selected, _, _, _) = model.forward(data["train"][0])
                shares = np.bincount(selected.ravel(), minlength=converted.experts) / selected.size
                close(shares, s["final_training_assignment_share"], "expert shares")
                assert np.all(shares > 0)
                for expert in range(converted.experts):
                    change = np.linalg.norm(model.p["W"][expert] - converted.p["W"][expert])
                    assert change > 0
                    close(change, s["expert_weight_change_l2"][expert], "expert changed")
                change = np.linalg.norm(model.p["R"] - converted.p["R"])
                assert change > 0
                close(change, s["router_weight_change_l2"], "router changed")
        budget = s["branches"]["linear_mac_matched"]["forward_projection_macs"]
        target = s["branches"]["moe"]["forward_projection_macs"]
        assert budget >= target and budget - target < len(data["train"][1]) * s["linear_projection_macs_per_sample"]
    compare_evidence(aggregate_summaries(summaries), read_json(folder / "aggregate.json"))
    with (folder / "comparison.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(summaries) * 5
    for row in rows:
        s = next(s for s in summaries if (s["split_seed"], s["seed"]) == (int(row["split_seed"]), int(row["seed"])))
        b = s["branches"][row["model"]]
        for split in ("train", "val", "test"):
            close(float(row[split + "_ce"]), b[split]["ce"], "comparison CSV")
    return cfg, summaries


def experiment_arguments(cfg):
    return dict(seeds=cfg["seeds"], split_seeds=cfg["split_seeds"],
        pre_epochs=cfg["pretrain_epochs"], post_epochs=cfg["continuation_epochs"],
        experts=cfg["experts"], top_k=cfg["top_k"], batch_size=cfg["batch_size"], alpha=cfg["balance_alpha"])


def reproduce(cfg, expected):
    with tempfile.TemporaryDirectory(prefix="linear-moe-reproduction-") as folder:
        actual, _ = run_experiment(folder, verbose=False, **experiment_arguments(cfg))
        compare_evidence(actual, expected)
    print(f"PASS: fresh script reproduction of {len(expected)} cases and all five branches.")


def check_singleton_plot():
    from plots import plot_results
    with tempfile.TemporaryDirectory(prefix="linear-moe-singleton-") as folder:
        run_experiment(folder, seeds=(8,), split_seeds=(2030,), verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            paths = plot_results(folder)
        assert len(paths) == 5 and all(p.stat().st_size > 1000 for p in paths)
    print("PASS: singleton plots with zero SD and no numerical warnings.")


def check_kernel(expected):
    import nbformat
    from notebook_tools import execute_notebook
    nb = nbformat.read(ROOT / "linear_to_moe.ipynb", as_version=4)
    for cell in nb.cells:
        if cell.cell_type == "code":
            cell.execution_count, cell.outputs = None, []
            cell.metadata.pop("execution", None)
    with tempfile.TemporaryDirectory(prefix="linear-moe-jupyter-check-") as folder:
        execute_notebook(nb, folder)
        compare_evidence(read_json(Path(folder) / "results/summary.json"), expected)
    print("PASS: fresh Jupyter kernel executed every notebook cell and reproduced all results.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--execute-notebook", action="store_true")
    parser.add_argument("--check-singleton", action="store_true")
    args = parser.parse_args()
    print(f"PASS: {check_manifest()} file-integrity checks.")
    print(f"PASS: {check_local_links()} local documentation links.")
    print("PASS: saved notebook execution:", check_notebook())
    with threadpool_limits(limits=1):
        print("PASS: numerical gradients, sparse dispatch, exact resumption and tolerance regression.", verify())
        cfg, summaries = check_results()
        print(f"PASS: {len(summaries)} cases: checkpoints, data splits, every branch, raw CSVs, routing, budgets and aggregates.")
        if args.reproduce:
            reproduce(cfg, summaries)
        if args.check_singleton:
            check_singleton_plot()
    if args.execute_notebook:
        check_kernel(summaries)
    print("Submission verification complete.")


if __name__ == "__main__":
    main()
