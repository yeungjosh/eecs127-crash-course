"""Smoke tests: every example script should run end to end without errors.

Run from the repo root with:  pytest tests/ -v
Plots use the non-interactive Agg backend, so no display is needed.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = sorted(ROOT.glob("*/examples/*.py"))


def test_example_scripts_found():
    assert SCRIPTS, "expected at least one example script under */examples/"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: str(p.relative_to(ROOT)))
def test_example_runs(script):
    env = dict(os.environ, MPLBACKEND="Agg")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
