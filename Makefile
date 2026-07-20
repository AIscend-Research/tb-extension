.PHONY: setup smoke test manifest loco clean

setup:
	bash setup.sh

smoke:
	python scripts/smoke_test.py

test:
	pytest -q

manifest:
	python scripts/build_manifest.py --data-root data

# full leave-one-cohort-out sweep with the baseline config
loco:
	python scripts/run_loco.py --config configs/base.yaml

# same sweep with CORAL, to compare against the baseline
loco-coral:
	python scripts/run_loco.py --config configs/coral.yaml

clean:
	rm -rf .venv runs __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
