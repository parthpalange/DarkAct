#!/usr/bin/env bash
set -e

echo "=== Setting up DarkAct Python Virtual Environment (venv) ==="

ENV_DIR="darkact_env"

if [ ! -d "$ENV_DIR" ]; then
    echo "Creating virtual environment in ./$ENV_DIR ..."
    python3 -m venv "$ENV_DIR"
else
    echo "Virtual environment ./$ENV_DIR already exists."
fi

echo "Activating virtual environment..."
source "$ENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing PyTorch with CUDA support..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Installing project requirements..."
pip install -r requirements.txt

echo "=== Verification ==="
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
"

echo "=== Setup complete! ==="
echo "To activate this environment anytime, run:"
echo "    source $ENV_DIR/bin/activate"
