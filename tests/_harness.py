"""A tiny test harness, so the suite runs with plain Python.

pytest is not installed in the environment this project uses, and installing it
would mean modifying the advisor's virtualenv. The tests matter more than the
runner, so this provides just enough of pytest's surface -- `parametrize` and a
discovery loop -- to run them with no setup at all.

If pytest is ever installed, the test files are written so it can collect them
unchanged: same `test_*` naming, same `parametrize(argnames, argvalues)` shape.
"""
from __future__ import annotations

import traceback
from typing import Callable

_PARAM_ATTR = "_params"


def parametrize(argnames: str, argvalues: list) -> Callable:
    """Attach parameter sets to a test function.

    `argnames` is a comma-separated string, matching pytest's signature, so a
    decorated test reads identically under either runner.
    """
    names = [n.strip() for n in argnames.split(",")]

    def decorator(func: Callable) -> Callable:
        cases = []
        for value in argvalues:
            values = value if isinstance(value, tuple) else (value,)
            cases.append(dict(zip(names, values)))
        setattr(func, _PARAM_ATTR, getattr(func, _PARAM_ATTR, []) + cases)
        return func

    return decorator


def run_module(namespace: dict, title: str = "tests") -> int:
    """Run every test_* callable in `namespace`. Returns an exit code."""
    tests = sorted(
        (name, obj)
        for name, obj in namespace.items()
        if name.startswith("test_") and callable(obj)
    )

    passed = failed = 0
    failures: list[tuple[str, str]] = []

    for name, func in tests:
        cases = getattr(func, _PARAM_ATTR, [{}])
        for case in cases:
            label = f"{name}{('[' + ', '.join(map(str, case.values())) + ']') if case else ''}"
            try:
                func(**case)
                passed += 1
            except Exception:
                failed += 1
                failures.append((label, traceback.format_exc()))

    for label, trace in failures:
        print(f"\nFAIL {label}\n{trace}")

    total = passed + failed
    print(f"\n{title}: {passed}/{total} passed" + (f", {failed} FAILED" if failed else ""))
    return 1 if failed else 0
