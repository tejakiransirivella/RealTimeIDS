#!/bin/bashc
SCRIPT_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
echo "Script directory: $SCRIPT_DIR"
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

create_venv() {
    echo "Creating virtual environment..."
    python3 -m venv "${PROJECT_ROOT}/venv"
    echo "Virtual environment created."
}

activate_venv() {
    echo "Activating virtual environment..."
    source "${PROJECT_ROOT}/venv/bin/activate"
    echo "Virtual environment activated."
}

echo "Checking for virtual environment..."
if [ -d "${PROJECT_ROOT}/venv" ]; then echo "venv exists"; 
else {
    echo "venv does not exist";
    create_venv
}
fi
activate_venv

echo "Installing dependencies..."
pip3 install -r $SCRIPT_DIR/requirements.txt
echo "Dependencies installed."