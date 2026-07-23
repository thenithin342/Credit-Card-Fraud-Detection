"""tests/unit/test_drift_monitor.py
────────────────────────────────────────────────────────────────────────
Unit tests for src/monitoring/drift_monitor.py.

What we verify
--------------
* ``summarize_drift`` extracts ``drift_share`` and ``drifted_columns`` from
  a raw Evidently JSON payload.
* ``run_monitor`` writes a JSON report to the configured directory and
  exits with code 0 when the drift share is below the threshold
  (synthetic identical reference and current → no drift).
* The script's CLI ``main`` returns 1 when drift is above the
  threshold (simulated via a tiny fake drift payload that
  ``summarize_drift`` would treat as a real detection).

The test never relies on a real ``test_features.parquet`` existing on
disk — both reference and current are built from ``pd.DataFrame`` and
plumbed through the public function directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture()
def params_yaml(tmp_path: Path) -> Path:
    """A minimal params.yaml with a writable report_output_dir."""
    p = tmp_path / "params.yaml"
    p.write_text(
        "\n".join(
            [
                "data:",
                "  features_dir: data/features",
                "monitoring:",
                "  drift_score_threshold: 0.5",
                "  reference_window_days: 30",
                "  report_output_dir: reports/evidently",
            ]
        )
    )
    return p


@pytest.fixture()
def reference_parquet(tmp_path: Path) -> Path:
    """A tiny synthetic numeric DataFrame as a parquet file."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "f1": rng.normal(0, 1, 200),
            "f2": rng.normal(5, 2, 200),
            "f3": rng.integers(0, 10, 200).astype(float),
        }
    )
    out = tmp_path / "test_features.parquet"
    df.to_parquet(out)
    return out


def _touch_features_dir(params_path: Path, parquet: Path) -> None:
    """Materialise the directory tree the script expects: <repo>/<features_dir>/<file>."""
    features_dir = params_path.parent / "data" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    target = features_dir / "test_features.parquet"
    target.write_bytes(parquet.read_bytes())


# ── unit: summarize_drift ───────────────────────────────────────────────


def test_summarize_drift_extracts_share_and_count() -> None:
    from src.monitoring.drift_monitor import summarize_drift

    raw = {
        "metrics": [
            {
                "metric_name": "DriftedColumnsCount(drift_share=0.5)",
                "config": {"type": "evidently:metric_v2:DriftedColumnsCount"},
                "value": {"count": 2.0, "share": 0.4},
                "id": "x",
            },
            {
                "metric_name": "ValueDrift(column=f1,method=K-S p_value,threshold=0.05)",
                "config": {"column": "f1", "method": "K-S p_value"},
                "value": 0.001,
                "id": "y",
            },
            {
                "metric_name": "ValueDrift(column=f2,method=K-S p_value,threshold=0.05)",
                "config": {"column": "f2", "method": "K-S p_value"},
                "value": 0.9,
                "id": "z",
            },
        ]
    }

    s = summarize_drift(raw)
    assert s["drift_share"] == pytest.approx(0.4)
    assert s["drifted_columns"] == 2
    cols = {c["column"]: c for c in s["per_column"]}
    assert cols["f1"]["drifted"] is True
    assert cols["f2"]["drifted"] is False


# ── end-to-end: identical reference/current → no drift → exit 0 ─────────


def test_run_monitor_no_drift_writes_report(
    params_yaml: Path, reference_parquet: Path
) -> None:
    from src.monitoring.drift_monitor import run_monitor

    _touch_features_dir(params_yaml, reference_parquet)

    summary = run_monitor(
        params_path=params_yaml,
        current_path=reference_parquet,  # identical → no drift
    )
    assert summary["drift_detected"] is False
    assert summary["drift_share"] == pytest.approx(0.0, abs=1e-3)
    assert summary["threshold"] == 0.5

    # The report file should exist.
    report_dir = params_yaml.parent / "reports" / "evidently"
    files = list(report_dir.glob("drift_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["drift_detected"] is False


def test_main_exits_nonzero_when_drift_detected(
    params_yaml: Path, reference_parquet: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force a high drift_share from the monitor and assert main() exits 1."""
    from src.monitoring import drift_monitor as mod

    def _fake_summary(*_args, **_kwargs) -> dict:
        return {
            "drift_share": 0.9,
            "drifted_columns": 9,
            "per_column": [],
            "threshold": 0.5,
            "drift_detected": True,
            "reference_rows": 200,
            "current_rows": 200,
            "generated_at": "1970-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(mod, "run_monitor", _fake_summary)

    rc = mod.main(["--params", str(params_yaml)])
    assert rc == 1


def test_main_exits_zero_when_no_drift(
    params_yaml: Path, reference_parquet: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.monitoring import drift_monitor as mod

    def _fake_summary(*_args, **_kwargs) -> dict:
        return {
            "drift_share": 0.05,
            "drifted_columns": 0,
            "per_column": [],
            "threshold": 0.5,
            "drift_detected": False,
            "reference_rows": 200,
            "current_rows": 200,
            "generated_at": "1970-01-01T00:00:00+00:00",
        }

    monkeypatch.setattr(mod, "run_monitor", _fake_summary)

    rc = mod.main(["--params", str(params_yaml)])
    assert rc == 0
