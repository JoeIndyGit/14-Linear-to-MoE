"""Numerical checks for handwritten gradients, conversion, and resumption."""
import copy
import json
from pathlib import Path
import tempfile
import numpy as np
from train import Linear, MoE, Adam, save_checkpoint, load_checkpoint
from evidence import compare_evidence


def check_gradients(model, x, y, alpha):
    _, grads, _ = model.loss_grad(x, y, alpha)
    worst = 0.0
    count = 0
    for name, parameter in model.p.items():
        for index in np.ndindex(parameter.shape):
            old = parameter[index]
            parameter[index] = old + 1e-6
            plus = model.loss_grad(x, y, alpha)[0]
            parameter[index] = old - 1e-6
            minus = model.loss_grad(x, y, alpha)[0]
            parameter[index] = old
            numeric = (plus - minus) / 2e-6
            analytic = grads[name][index]
            error = abs(numeric - analytic) / max(1.0, abs(numeric), abs(analytic))
            worst = max(worst, error)
            count += 1
    assert worst < 1e-7, worst
    return {"parameters_checked": count, "max_scaled_error": float(worst)}


def verify():
    rng = np.random.default_rng(90210)
    x, y = rng.normal(size=(11, 4)), rng.integers(0, 3, 11)
    linear = Linear(8, d=4, classes=3)
    moe = MoE(linear, 99)
    initial_task_gradient = np.linalg.norm(moe.loss_grad(x, y, 0.0)[1]["R"])
    assert initial_task_gradient < 1e-14
    scratch = MoE.from_scratch(8, d=4, classes=3)
    assert not np.array_equal(scratch.p["W"][0], scratch.p["W"][1])
    compare_evidence({"loss": 0.123456789 + 1e-9, "training_seconds": 3.0},
                     {"loss": 0.123456789, "training_seconds": 1.0})
    try:
        compare_evidence({"loss": 0.13}, {"loss": 0.12})
    except AssertionError:
        pass
    else:
        raise AssertionError("Numerical comparison accepted a material loss change.")
    delta = float(np.max(np.abs(linear.forward(x)[0] - moe.forward(x)[0])))
    assert delta < 1e-12
    assert not np.shares_memory(linear.p["W"], moe.p["W"])
    for a in range(4):
        for b in range(a + 1, 4):
            assert not np.shares_memory(moe.p["W"][a], moe.p["W"][b])
    # Break expert symmetry for a meaningful router task-gradient check.
    moe.p["W"] += rng.normal(0, 0.1, moe.p["W"].shape)
    moe.p["b"] += rng.normal(0, 0.1, moe.p["b"].shape)
    checks = {"linear_gradient": check_gradients(linear, x, y, 0.0),
              "moe_task_gradient": check_gradients(moe, x, y, 0.0),
              "moe_task_plus_balance_gradient": check_gradients(moe, x, y, 0.01),
              "conversion_max_abs_error": delta}
    logits, (selected, weights, _, _) = moe.forward(x)
    assert np.allclose(weights.sum(axis=1), 1)
    assert np.all(selected[:, 0] != selected[:, 1])
    assert selected.size == len(x) * 2
    # Independent dense reference is used ONLY to verify sparse dispatch.
    reference = np.zeros_like(logits)
    for i in range(len(x)):
        for slot, expert in enumerate(selected[i]):
            reference[i] += weights[i, slot] * (x[i] @ moe.p["W"][expert] + moe.p["b"][expert])
    assert np.allclose(logits, reference, atol=1e-14)
    optimizer = Adam(moe.p)
    optimizer.step(moe.p, moe.loss_grad(x, y)[1])
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "resume.npz"
        save_checkpoint(path, moe, optimizer, rng)
        restored, restored_opt, restored_rng, _ = load_checkpoint(path)
        assert np.array_equal(rng.permutation(11), restored_rng.permutation(11))
        optimizer.step(moe.p, moe.loss_grad(x, y)[1])
        restored_opt.step(restored.p, restored.loss_grad(x, y)[1])
        assert all(np.array_equal(moe.p[k], restored.p[k]) for k in moe.p)
    checks.update({"sparse_dispatch_reference": "passed", "normalized_top2_weights": "passed",
                   "independent_expert_storage": "passed", "checkpoint_exact_next_step": "passed"})
    checks.update({"copied_experts_initial_task_router_gradient_l2": float(initial_task_gradient),
                   "scratch_experts_independently_initialized": "passed",
                   "numerical_tolerance_accepts_roundoff_rejects_material_change": "passed"})
    return checks


if __name__ == "__main__":
    checks = verify()
    print(json.dumps(checks, indent=2))
    Path("results").mkdir(exist_ok=True)
    Path("results/verification.json").write_text(json.dumps(checks, indent=2))
