#!/usr/bin/env bash
# Downloads raw data into ./download
# and saves preprocessed data into ./data
# Get directory containing this script

# UPDATED FOR TINYIMAGENET

CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH=$PYTHONPATH:$CODE_DIR

pip install -r $CODE_DIR/requirements.txt

python $CODE_DIR/preprocessing/get_dataset.py

