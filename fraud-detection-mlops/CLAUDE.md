# CLAUDE.md — Instructions for Claude Code

This file is read by Claude Code at the start of every session in this repo.
It is the operating manual for building this project. Read it fully before
writing any code.

## 1. What this project is

A real-time fraud/anomaly detection platform built to demonstrate the full
MLOps lifecycle end-to-end: streaming ingestion, a hand-built feature store,
tracked model training, low-latency serving, drift detection, and automated
retraining — all running locally via Docker, with no paid API keys required.

**Read these before doing anything, in this order:**
1. `docs/PRD.md` — what we're building and why
2. `docs/TECH_STACK.md` — what tools, and the reasoning per tool
3. `docs/ARCHITECTURE.md` — how the pieces connect, with a diagram
4. `docs/ROADMAP.md` — the phased build plan with Definition of Done per phase
5. `docs/DATA_SPEC.md` — dataset, schema, and known limitations

## 2. How to work in this repo

- **Follow the roadmap in order.** Do not jump ahead to Phase 3 work while
  Phase 1 is incomplete. If asked to build something out of order, flag the
  roadmap dependency before proceeding.
- **Definition of Done is not optional.** Each phase in `docs/ROADMAP.md` has an
  explicit DoD. A phase isn't finished until its DoD is verifiably true (tests
  passing, containers healthy, etc.) — not "the code is written."
- **One phase, one focused session where possible.** Prefer finishing and
  verifying a phase over starting several in parallel.
- **Log architecture decisions.** Any time you make a non-trivial design choice
  (a library, a schema, a tradeoff), append an entry to `docs/DECISIONS.md`
  using the template already in that file. This is the single most valuable
  artifact for turning this project into interview answers — do not skip it.

## 3. Engineering conventions

- Python 3.11, type hints on all function signatures, Google-style docstrings
  on all public functions/classes.
- Formatting/linting: `black`, `ruff`, `mypy` — should pass via `pre-commit`
  before any commit is considered done.
- Every new module gets at least one corresponding test in `tests/`, mirroring
  the `src/` structure (e.g. `src/features/definitions.py` →
  `tests/features/test_definitions.py`).
- No hardcoded secrets or connection strings anywhere in code — everything
  through `.env` + `pydantic-settings`. Update `.env.example` whenever a new
  variable is introduced.
- Favor explicit, readable code over clever abstractions. The point of this
  project is that every line can be explained in an interview — don't reach
  for a framework feature you can't defend.
- Commit messages: conventional commits style (`feat:`, `fix:`, `docs:`,
  `test:`, `chore:`).

## 4. Hard constraints — do not violate these

- Everything must run via `docker-compose up` with **zero paid services and no
  API keys**, except the one-time Kaggle CLI download of the dataset (which is
  free and documented in `docs/DATA_SPEC.md`).
- Do not introduce a managed cloud dependency (no AWS/GCP/Azure-only services)
  for the core deliverable. Kubernetes/cloud is explicitly Phase 8/stretch only.
- Do not replace the intentionally hand-built components (feature store, drift
  detector — see `docs/TECH_STACK.md` §"Design principle") with a managed
  library (Feast, Evidently) without flagging it as a scope change first — that
  substitution would remove the interview-differentiating value of this project.
- Never commit the raw dataset directly to git — it must be DVC-tracked.
- If a task requires installing something outside PyPI/npm/GitHub, or requires
  network access to a domain that isn't already in this environment's allowed
  list, stop and ask before proceeding.

## 5. Definition of "done" for the whole project

Pulled directly from `docs/PRD.md` §6 — the finished system should:
- Score transactions at p95 < 150ms
- Show zero train/serve feature skew (automated parity test passing)
- Correctly flag injected synthetic drift in a controlled test
- Run the full pipeline via `docker-compose up` with zero manual steps
- Pass CI (lint + test + build) on every commit
- Execute a full retraining cycle from a drift trigger to a reviewable
  candidate model in the MLflow registry

## 6. First task — start here

Run Phase 0 from `docs/ROADMAP.md`. A ready-to-paste prompt for this is in
`README.md` under "Where to start." After finishing Phase 0, stop, summarize
what changed, confirm the Definition of Done checklist, and wait for
confirmation before starting Phase 1.
