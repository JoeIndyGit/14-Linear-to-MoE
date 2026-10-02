"""Compare numerical evidence with explicit tolerances, excluding run timings."""
from numbers import Real
import numpy as np

TIMING_KEYS = frozenset({"training_seconds", "elapsed_seconds"})


def compare_evidence(actual, expected, *, rtol=1e-6, atol=1e-8, path="root"):
    if isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), f"{path}: dictionary keys differ"
        for key in expected.keys() - TIMING_KEYS:
            compare_evidence(actual[key], expected[key], rtol=rtol, atol=atol, path=f"{path}/{key}")
    elif isinstance(expected, (list, tuple)):
        assert isinstance(actual, (list, tuple)) and len(actual) == len(expected), f"{path}: length differs"
        for i, (new, old) in enumerate(zip(actual, expected)):
            compare_evidence(new, old, rtol=rtol, atol=atol, path=f"{path}/{i}")
    elif isinstance(expected, Real) and not isinstance(expected, (bool, int)):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=path)
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"
