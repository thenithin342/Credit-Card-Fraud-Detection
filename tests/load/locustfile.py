"""tests/load/locustfile.py
─────────────────────────────────────────────────────────────────────────────
Locust load test for the FraudGuard scoring API.

Target endpoint
---------------
``POST /v1/score`` — the real-time fraud-scoring route defined in
``src/serving/app.py`` and validated by ``tests/unit/test_serving.py``.

Latency SLA
-----------
p95 latency must stay **below 150 ms** (see
``src.config.Settings.scoring_latency_sla_ms``).  Per-request latency is
checked against that threshold and any over-budget request is marked as
a Locust failure so the breach shows up in the report — not just in a
latency column.

Run
---
Headless smoke:

    locust -f tests/load/locustfile.py \\
           --host=http://localhost:8000 \\
           --users 10 --spawn-rate 2 --run-time 30s --headless

Full SLA run (matches the spec in the Phase 3 closeout plan):

    locust -f tests/load/locustfile.py \\
           --host=http://localhost:8000 \\
           --users 50 --spawn-rate 5 --run-time 60s --headless

Interactive (web UI at http://localhost:8089):

    locust -f tests/load/locustfile.py --host=http://localhost:8000

Dependencies
------------
``locust>=2.28,<3.0`` is already pinned in ``requirements-dev.txt``.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

from locust import HttpUser, between, task

# Ensure src is importable when locust is run from any directory
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
try:
    from src.config import get_settings

    _SLA_MS = float(get_settings().scoring_latency_sla_ms)
except Exception:
    _SLA_MS = 150.0  # fallback to PRD default

# IEEE-CIS reference epoch (matches ``tests/conftest.py`` and
# ``tests/unit/test_serving.py``).  90 simulated days on top of that.
_EPOCH_START = 86_400
_EPOCH_SPAN_SECONDS = 90 * 86_400


def _sample_payload(rng: random.Random) -> dict[str, object]:
    """Build a single synthetic ``TransactionRequest`` payload.

    Mirrors the IEEE-CIS column shape used by the preprocessor (see
    ``src/features/build_features.py``).  Only the four required-ish
    fields are populated deterministically; a handful of common raw
    columns are added so the request looks like real traffic.  Any
    field the preprocessor doesn't recognise is dropped silently,
    so the test is robust to schema drift.
    """
    # Log-uniform amount in [1, 2000] — real card transactions are
    # heavily right-skewed, and a uniform spread would mask tail latency.
    amount = round(math.exp(rng.uniform(0.0, math.log(2000.0))), 2)

    return {
        "transaction_id": rng.randint(1_000_000, 9_999_999),
        "TransactionDT": rng.randint(_EPOCH_START, _EPOCH_START + _EPOCH_SPAN_SECONDS),
        "TransactionAmt": amount,
        "card1": rng.randint(1000, 9999),
        # Optional raw columns — preprocessor ignores unknowns.
        "ProductCD": rng.choice(["W", "H", "C", "S", "R"]),
        "card4": rng.choice(["visa", "mastercard", "discover", "american express"]),
        "card6": rng.choice(["credit", "debit"]),
        "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", "hotmail.com"]),
    }


class FraudScoringUser(HttpUser):
    """A simulated card-holder driving the scoring API.

    ``weight = 1`` so a plain ``locust -f locustfile.py`` invocation
    (no ``--users`` flag) ramps to 10 default users, matching the
    Phase 3 spec.
    """

    weight = 1
    # Tight inter-arrival jitter so each user produces ~5–20 rps,
    # enough to put real pressure on the 150 ms p95 budget without
    # overwhelming a single-process uvicorn worker.
    wait_time = between(0.05, 0.2)

    def on_start(self) -> None:
        # Per-user RNG keeps each simulated user emitting a stable
        # distribution without locking out the global random state.
        self._rng = random.Random()

    @task
    def score_transaction(self) -> None:
        payload = _sample_payload(self._rng)
        with self.client.post(
            "/v1/score",
            json=payload,
            name="POST /v1/score",  # group stats under one row in the report
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(
                    f"status={resp.status_code} body={resp.text[:200]!r}"
                )
                return

            latency_ms = resp.elapsed.total_seconds() * 1000.0
            if latency_ms > _SLA_MS:
                # Per-request SLA breach — surfaces in the report's
                # failure column so the run fails loudly instead of
                # quietly creeping over the p95 budget.
                resp.failure(
                    f"p95 SLA breach: {latency_ms:.1f}ms > {_SLA_MS:.0f}ms"
                )
            else:
                resp.success()
