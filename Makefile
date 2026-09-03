# vid2spatial_v2 — test / contract entry points (CPU only; no models, no CUDA)
PY ?= python3
export CUDA_VISIBLE_DEVICES :=

.PHONY: test contract-check contract-regen lint check-engine attach-dry-run verify-engine-audio

test: contract-check
	$(PY) -m pytest test/test_unit.py test/test_integration.py test/test_bridge_contract.py -q

contract-check:
	$(PY) tools/extract_bridge_contract.py --check

contract-regen:
	$(PY) tools/extract_bridge_contract.py

lint:
	ruff check tools/extract_bridge_contract.py test/test_bridge_contract.py tools/attach_engine.py

# Attach-readiness. contract-check pins the wire against the engine lane ref
# (V2S_BRIDGE_REF, default fix/lane-bridge-handoff).
check-engine:
	$(PY) tools/attach_engine.py --check-engine --host $(ENGINE_HOST) --port $(ENGINE_PORT)

attach-dry-run:
	$(PY) tools/attach_engine.py $(TRAJ) --dry-run --limit 5

# Offline proof that the engine actually renders audio (needs a lane binary).
# Also the isolation matrix for "the engine is silent" reports.
verify-engine-audio:
	bash tools/repro_engine_silence.sh

ENGINE_HOST ?= 127.0.0.1
ENGINE_PORT ?= 9000
