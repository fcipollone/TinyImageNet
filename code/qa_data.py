from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import tarfile
import argparse

from six.moves import urllib

from tqdm import *
from os.path import join as pjoin

# Preprocessing and data tools

def setup_args():
    parser = argparse.ArgumentParser()
    code_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    glove_dir = os.path.join("download", "dwr")
    source_dir = os.path.join("data", "tiny-imagenet-200")
    parser.add_argument("--source_dir", default=source_dir)

    return parser.parse_args()


if __name__ == '__main__':
    args = setup_args()
