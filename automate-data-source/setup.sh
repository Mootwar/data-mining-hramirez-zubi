#!/usr/bin/env bash
# Exit immediately if a command fails, treat unset variables as errors, and fail pipelines if any command fails
set -euo pipefail

# Check Python 3
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is not installed or not on PATH."
  echo "Install it (e.g., Debian/Ubuntu: sudo apt-get install python3)"
  exit 1
fi

# Check venv module
if ! python3 -c 'import venv' >/dev/null 2>&1; then
  echo "Error: Python venv module is missing."
  echo "Install it (e.g., Debian/Ubuntu: sudo apt-get install python3-venv)"
  exit 1
fi

echo "OK: $(python3 --version) with venv is available."

# Create venv for project
python3 -m venv .venv

source .venv/bin/activate

pip install -r 'requirements.txt'

echo "OK: Virtual environment created and dependencies installed."