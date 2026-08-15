.PHONY: help setup smoke data manifest train eval test lint clean

CONFIG ?= configs/loco_montgomery.yaml
CLINIC ?= montgomery

help:
	@echo "make setup     - create venv and install (editable) with dev+data extras"
	@echo "make smoke     - run the no-data smoke test"
	@echo "make data      - download the aggregated Kaggle TB set (needs kaggle token)"
	@echo "make manifest  - build data/processed/manifest.csv and print class balance"
	@echo "make train     - train (CONFIG=... to pick an experiment)"
	@echo "make eval      - evaluate (CONFIG=... CLINIC=... for the checkpoint path)"
	@echo "make test      - run pytest"
	@echo "make lint      - ruff check"
	@echo "make clean     - remove caches and build artifacts"

setup:
	python -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -e ".[dev,data]"
	@echo "Activate with: source .venv/bin/activate"

smoke:
	python scripts/smoke_test.py

data:
	python scripts/download_data.py --kaggle-aggregated

manifest:
	python scripts/build_manifest.py --raw data/raw --out data/processed/manifest.csv

train:
	tbtrust-train --config $(CONFIG)

eval:
	tbtrust-eval --config $(CONFIG) --checkpoint outputs/$(CLINIC)/best.ckpt

test:
	pytest -q

lint:
	ruff check src scripts tests

clean:
	rm -rf __pycache__ */__pycache__ **/__pycache__ .pytest_cache .ruff_cache build *.egg-info
	find . -name "*.pyc" -delete
