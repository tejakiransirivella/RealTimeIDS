SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

echo "Installing dependencies..."
pip3 install -r $SCRIPT_DIR/requirements.txt
echo "Dependencies installed."

echo "Project root: $PROJECT_ROOT"

CONFIG_PATH="$PROJECT_ROOT/config/path.json"

jq --arg project_root "$PROJECT_ROOT" '.project_path = $project_root' "$CONFIG_PATH" > temp.json && mv temp.json "$CONFIG_PATH"

data="$1"
python3 $PROJECT_ROOT/src/python/sniff.py "$data"








