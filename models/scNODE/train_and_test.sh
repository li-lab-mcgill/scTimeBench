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
echo "Activating scNODE virtual environment..."
source ./models/scNODE/scNODE_module/.venv/bin/activate
echo "Virtual environment activated."

# 6. Now let's run train and test on model.py with the provided YAML file
python ./models/scNODE/run.py --yaml_config "$FILE_PATH"
