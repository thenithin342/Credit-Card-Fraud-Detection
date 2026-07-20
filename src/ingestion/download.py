"""
src/ingestion/download.py
─────────────────────────────────────────────────────────────────────────────
Kaggle dataset downloader for FraudGuard.

Downloads two datasets:
  1. IEEE-CIS Fraud Detection  →  data/raw/ieee-cis/
  2. ULB Credit Card Fraud     →  data/raw/ulb/

Usage:
    python -m src.ingestion.download              # download both
    python -m src.ingestion.download --dataset ieee   # only IEEE-CIS
    python -m src.ingestion.download --dataset ulb    # only ULB
    python -m src.ingestion.download --check          # verify files exist

Requirements:
    - KAGGLE_USERNAME and KAGGLE_KEY set in .env (or ~/.kaggle/kaggle.json)
    - pip install kaggle
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

import structlog
from dotenv import load_dotenv

load_dotenv()

log = structlog.get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASETS = {
    "ieee": {
        "competition": "ieee-fraud-detection",
        "out_dir": PROJECT_ROOT / "data" / "raw" / "ieee-cis",
        "expected_files": [
            "train_transaction.csv",
            "train_identity.csv",
            "test_transaction.csv",
            "test_identity.csv",
        ],
    },
    "ulb": {
        "competition": None,
        "dataset": "mlg-ulb/creditcardfraud",
        "out_dir": PROJECT_ROOT / "data" / "raw" / "ulb",
        "expected_files": ["creditcard.csv"],
    },
}


# ── Helpers ────────────────────────────────────────────────────────────────


def _check_kaggle_credentials() -> None:
    """
    Resolve Kaggle credentials from environment variables or ~/.kaggle/ files.

    Supports two formats:
      NEW (post-2024): KAGGLE_API_TOKEN=KGAT_...  → written to ~/.kaggle/access_token
      OLD (classic):   KAGGLE_USERNAME + KAGGLE_KEY → written to ~/.kaggle/kaggle.json
    """
    import json

    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    access_token_file = kaggle_dir / "access_token"

    # ── NEW format: single KAGGLE_API_TOKEN ───────────────────────────────
    api_token = os.getenv("KAGGLE_API_TOKEN")
    if api_token and api_token.startswith("KGAT_"):
        if not access_token_file.exists():
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            access_token_file.write_text(api_token)
            try:
                access_token_file.chmod(0o600)
            except Exception:
                pass  # chmod may fail on Windows; not critical
            log.info("kaggle_access_token_written", path=str(access_token_file))
        return  # credentials resolved

    # ── OLD format: KAGGLE_USERNAME + KAGGLE_KEY ──────────────────────────
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if username and key:
        if not kaggle_json.exists():
            kaggle_dir.mkdir(parents=True, exist_ok=True)
            kaggle_json.write_text(json.dumps({"username": username, "key": key}))
            try:
                kaggle_json.chmod(0o600)
            except Exception:
                pass
            log.info("kaggle_json_written", path=str(kaggle_json))
        return  # credentials resolved

    # ── Already on disk ───────────────────────────────────────────────────
    if kaggle_json.exists() or access_token_file.exists():
        return  # credentials already present

    # ── Nothing found ─────────────────────────────────────────────────────
    log.error(
        "kaggle_credentials_missing",
        hint=(
            "Set KAGGLE_API_TOKEN=KGAT_... in your .env file "
            "(new format), or place kaggle.json in ~/.kaggle/"
        ),
    )
    sys.exit(1)


def _unzip_and_clean(zip_path: Path, out_dir: Path) -> None:
    """Unzip *zip_path* into *out_dir* and remove the zip file."""
    log.info("unzipping", zip=str(zip_path), dest=str(out_dir))
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Some competition zips nest data in sub-directories; extract flat.
        for member in zf.namelist():
            filename = Path(member).name
            if not filename:
                continue
            source = zf.open(member)
            target = out_dir / filename
            with open(target, "wb") as f:
                shutil.copyfileobj(source, f)
    zip_path.unlink()
    log.info("zip_removed", zip=str(zip_path))


def _verify_files(dataset_key: str) -> bool:
    """Return True if all expected files exist for *dataset_key*."""
    cfg = DATASETS[dataset_key]
    out_dir: Path = cfg["out_dir"]
    missing = [f for f in cfg["expected_files"] if not (out_dir / f).exists()]
    if missing:
        log.warning("missing_files", dataset=dataset_key, files=missing)
        return False
    log.info("files_ok", dataset=dataset_key, dir=str(out_dir))
    return True


# ── Download Logic ─────────────────────────────────────────────────────────


def download_ieee() -> None:
    """Download the IEEE-CIS Fraud Detection competition dataset."""
    cfg = DATASETS["ieee"]
    out_dir: Path = cfg["out_dir"]

    if _verify_files("ieee"):
        log.info("ieee_already_downloaded", dir=str(out_dir))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading_ieee_cis", dest=str(out_dir))

    # kaggle library must be imported after credentials are resolved
    import kaggle  # noqa: F401 — triggers auth check

    from kaggle.api.kaggle_api_extended import KaggleApiExtended

    api = KaggleApiExtended()
    api.authenticate()

    # Download all competition files as a zip into out_dir
    api.competition_download_files(
        competition=cfg["competition"],
        path=str(out_dir),
        quiet=False,
    )

    # The download produces <competition>.zip — unzip it
    zip_file = out_dir / f"{cfg['competition']}.zip"
    if zip_file.exists():
        _unzip_and_clean(zip_file, out_dir)

    # Some files may be individually zipped (.csv.zip); expand them too
    for extra_zip in out_dir.glob("*.zip"):
        _unzip_and_clean(extra_zip, out_dir)

    if not _verify_files("ieee"):
        log.error("ieee_download_incomplete")
        sys.exit(1)

    log.info("ieee_download_complete", dir=str(out_dir))


def download_ulb() -> None:
    """Download the ULB Credit Card Fraud dataset."""
    cfg = DATASETS["ulb"]
    out_dir: Path = cfg["out_dir"]

    if _verify_files("ulb"):
        log.info("ulb_already_downloaded", dir=str(out_dir))
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("downloading_ulb", dest=str(out_dir))

    from kaggle.api.kaggle_api_extended import KaggleApiExtended

    api = KaggleApiExtended()
    api.authenticate()

    api.dataset_download_files(
        dataset=cfg["dataset"],
        path=str(out_dir),
        unzip=True,
        quiet=False,
    )

    if not _verify_files("ulb"):
        log.error("ulb_download_incomplete")
        sys.exit(1)

    log.info("ulb_download_complete", dir=str(out_dir))


# ── CLI ────────────────────────────────────────────────────────────────────


def main() -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ]
    )

    parser = argparse.ArgumentParser(description="Download FraudGuard datasets from Kaggle")
    parser.add_argument(
        "--dataset",
        choices=["ieee", "ulb", "all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify that expected files exist; do not download",
    )
    args = parser.parse_args()

    if args.check:
        targets = ["ieee", "ulb"] if args.dataset == "all" else [args.dataset]
        ok = all(_verify_files(t) for t in targets)
        sys.exit(0 if ok else 1)

    _check_kaggle_credentials()

    if args.dataset in ("ieee", "all"):
        download_ieee()
    if args.dataset in ("ulb", "all"):
        download_ulb()

    log.info("all_downloads_complete")


if __name__ == "__main__":
    main()
