#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="python"

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if ! "$PYTHON_BIN" -c "import pytest" >/dev/null 2>&1; then
  echo "pytest is not installed in: $PYTHON_BIN"
  echo "Install it with: $PYTHON_BIN -m pip install pytest"
  exit 1
fi

if [[ $# -eq 0 ]]; then
  TARGETS=("tests")
elif [[ "$1" == "unit" ]]; then
  TARGETS=("tests/unit")
elif [[ "$1" == "component" ]]; then
  TARGETS=("tests/component")
elif [[ "$1" == "integration" ]]; then
  TARGETS=("tests/integration")
elif [[ "$1" == "all" ]]; then
  TARGETS=("tests")
else
  TARGETS=("$@")
fi

echo "Using python: $PYTHON_BIN"
echo "PYTHONPATH: $PYTHONPATH"
echo "Running: $PYTHON_BIN -m pytest -q ${TARGETS[*]}"

"$PYTHON_BIN" -m pytest -q "${TARGETS[@]}"
