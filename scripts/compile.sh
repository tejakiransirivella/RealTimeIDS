SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
echo "Project root: $PROJECT_ROOT"
SRC_PATH="$PROJECT_ROOT/src/dt"
SRC_FILES="$SRC_PATH/decision_tree.cc $SRC_PATH/decision_tree.cu $SRC_PATH/bindings.cc"

echo "Compiling files from: $SRC_PATH"
echo "Source files: $SRC_FILES"

BUILD_DIR="$PROJECT_ROOT/build"

nvcc $SRC_FILES \
-Xcompiler "-O3 -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) -fopenmp -C" \
-o "${BUILD_DIR}/decision_tree$(python3-config --extension-suffix)" -code=sm_61 -arch=compute_61

echo "Compiled to: ${BUILD_DIR}/decision_tree$(python3-config --extension-suffix)"


