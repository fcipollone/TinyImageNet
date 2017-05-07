#!/usr/bin/env bash
# Downloads raw data into ./download
# and saves preprocessed data into ./data
# Get directory containing this script

# UPDATED FOR TINYIMAGENET

CODE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH=$PYTHONPATH:$CODE_DIR

pip install -r $CODE_DIR/requirements.txt

# download punkt, perluniprops
if [ ! -d "/usr/local/share/nltk_data/tokenizers/punkt" ]; then
    python -m nltk.downloader punkt
fi

DATA_DIR=data
DOWNLOAD_DIR=download
mkdir -p $DATA_DIR
rm -rf $DATA_DIR

# Download distributed word representations
python $CODE_DIR/preprocessing/get_dataset.py

# Data processing for TensorFlow
python $CODE_DIR/qa_data.py
