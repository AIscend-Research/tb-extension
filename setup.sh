#!/usr/bin/env bash
# One-shot local setup. Creates a venv, installs xctb, runs the smoke test.
# On Kaggle you do NOT need this; see README "Running on Kaggle".
set -euo pipefail

PYTHON="${PYTHON:-python3}"

echo ">> creating virtualenv in .venv"
"$PYTHON" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo ">> upgrading pip"
pip install --upgrade pip >/dev/null

echo ">> installing core dependencies"
pip install -r requirements.txt

echo ">> installing xctb (editable)"
pip install -e .

echo ">> running the torch-free smoke test"
python scripts/smoke_test.py

echo
echo ">> done. Activate the env with:  source .venv/bin/activate"
echo ">> next: put data under data/ (see docs/DATA.md), then"
echo "         python scripts/build_manifest.py --data-root data"
