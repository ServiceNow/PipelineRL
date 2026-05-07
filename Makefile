.PHONY: help install ci-install update lint xray

help:
	@echo "make install    - Install dependencies in editable mode"
	@echo "make ci-install - Install dependencies with locked versions (for CI)"
	@echo "make update     - Update dependencies"
	@echo "make xray       - Run Web UI to visualize rollouts (pass RES to specify results file)"

xray:
	@echo "🔍 Running results viewer on port 8765"
	uv run python -m results_viewer --results $(RES) --host 127.0.0.1 --port 8765

install:
	@echo "🚀 Installing dependencies"
# 	@echo "Install requires sudo permissions to install Playwright dependencies. You may be prompted for your password."
	uv sync --all-extras
# 	uv run playwright install chromium --with-deps
# 	git config core.hooksPath .githooks

ci-install:
	uv sync --frozen --all-extras
# 	uv run playwright install chromium --with-deps

update:
	@echo "🔄 Updating dependencies"
	uv sync --all-extras --upgrade
# 	uv run playwright install chromium --with-deps

lint:
	@echo "🧹 Linting code"
	uv run ruff check --fix .
	uv run ruff format .

lint-check:
	@echo "🧹 Checking lint"
	uv run ruff check --diff .
	uv run ruff format --diff .