# vid2spatial_v2 — test / contract entry points (CPU only; no models, no CUDA)
PY ?= python3
export CUDA_VISIBLE_DEVICES :=

.PHONY: test contract-check contract-regen lint

test: contract-check
	$(PY) -m pytest test/test_unit.py test/test_integration.py test/test_bridge_contract.py -q

contract-check:
	$(PY) tools/extract_bridge_contract.py --check

contract-regen:
	$(PY) tools/extract_bridge_contract.py

lint:
	ruff check tools/extract_bridge_contract.py test/test_bridge_contract.py
