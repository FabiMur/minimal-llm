#!/bin/bash
set -e
ENV_NAME="minimal-llm"
BASE="$(conda info --base)"
source "$BASE/etc/profile.d/conda.sh"

# Choose environment file based on OS
UNAME_S="$(uname -s)";  # Get OS name
UNAME_M="$(uname -m)";  # Get machine hardware name
if [[ "$UNAME_S" == "Linux" ]]; then
  ENV_FILE="environment.cuda.yaml"
elif [[ "$UNAME_S" == "Darwin" && "$UNAME_M" == "arm64" ]]; then
  ENV_FILE="environment.metal.yaml"
else
  echo "Unsupported platform: $UNAME_S $UNAME_M"; exit 1
fi

# Create or update the environment
if conda env list | grep -q "^$ENV_NAME "; then
  conda env update -f "$ENV_FILE" --prune
else
  conda env create -f "$ENV_FILE"
fi

# Activate the environment
conda activate "$ENV_NAME"

# Create Jupyter kernel
python -m ipykernel install --user --name="$ENV_NAME" --display-name "Python ($ENV_NAME)"

# Install pre-commit hooks in the repository
echo "Setting up pre-commit hooks..."
pre-commit install || true  # Ignore errors
echo "Pre-commit hooks installed."

echo "Environment ready: $ENV_NAME"
