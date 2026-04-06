#!/bin/bash

# 1. Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "Error: No file path provided."
    echo "Usage: $0 path/to/file.yaml"
    exit 1
fi

FILE_PATH=$1

# 2. Check if the file exists
if [ ! -f "$FILE_PATH" ]; then
    echo "Error: File '$FILE_PATH' not found."
    exit 1
fi

# 3. Validate file extension (optional but recommended)
extension="${FILE_PATH##*.}"
if [ "$extension" != "yaml" ] && [ "$extension" != "yml" ]; then
    echo "Warning: File does not have a .yaml or .yml extension."
fi

# 4. Success logic
echo "Success: Found $FILE_PATH"
echo "Processing content..."

# 5. Let's get the venv ready
echo "Activating ot_cfm virtual environment..."
VENV_DIR="./venv/ot_cfm"

if [ -d "$VENV_DIR" ] && [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Found virtualenv at $VENV_DIR. Activating..."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
else
    echo "Virtualenv not found at $VENV_DIR. Creating..."
    python3 -m venv "$VENV_DIR"
    if [ -f "$VENV_DIR/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "$VENV_DIR/bin/activate"
    else
        echo "Error: failed to create virtualenv at $VENV_DIR"
        exit 1
    fi

    echo "Upgrading pip and installing ot_cfm into the virtualenv..."
    pip install --upgrade pip
    if [ -f "./methods/ot_cfm/ot_cfm_module/requirements.txt" ]; then
        pip install -r ./methods/ot_cfm/ot_cfm_module/requirements.txt
    else
        echo "Warning: ot_cfm requirements.txt not found at ./methods/ot_cfm/ot_cfm_module/requirements.txt"
    fi
    pip install -e ./methods/ot_cfm/ot_cfm_module # makes ot_cfm accessible in the venv
    pip install -e . # makes scTimeBench accessible in the venv
fi

echo "Virtual environment activated."

# 6. Now let's run train and test on method.py with the provided YAML file
python ./methods/ot_cfm/run.py --yaml_config "$FILE_PATH"
