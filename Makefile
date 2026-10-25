PYTHON ?= python3
WHEEL_DIR ?= dist
TORCH_CPU_INDEX ?= https://download.pytorch.org/whl/cpu

.PHONY: wheel install test smoke docker clean

wheel:
	PIP_EXTRA_INDEX_URL="$(TORCH_CPU_INDEX)" $(PYTHON) -m pip wheel . --no-deps --wheel-dir "$(WHEEL_DIR)"

install: wheel
	$(PYTHON) -m pip install --force-reinstall torch==2.8.0 --index-url "$(TORCH_CPU_INDEX)"
	$(PYTHON) -m pip install --no-deps --force-reinstall "$(WHEEL_DIR)"/moe_topk-*.whl

test: install
	$(PYTHON) -m pip install pytest
	@temp_dir=$$(mktemp -d); trap 'rm -rf "$$temp_dir"' EXIT; \
	python_bin=$$($(PYTHON) -c 'import sys; print(sys.executable)'); \
	cp -R tests "$$temp_dir/tests" && cd "$$temp_dir" && \
	env -u PYTHONPATH "$$python_bin" -m pytest -q tests

smoke:
	PYTHON="$(PYTHON)" bash tools/smoke.sh

docker:
	docker build --tag moe-topk:local .
	docker run --rm moe-topk:local

clean:
	rm -rf build dist .pytest_cache .ruff_cache
	rm -rf ./*.egg-info src/*.egg-info
	rm -f src/moe_ops/_C*.so src/moe_ops/_C*.pyd
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
