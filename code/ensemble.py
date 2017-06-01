import io
import os
import json
import sys
import random
import re
from os.path import join as pjoin
import glob
from collections import defaultdict

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

from preprocessing.maybe_download import maybe_download
from ti_model import Model
from ti_classifiers import get_classifier
from utils import *
from data_utils import *

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_string("train_dir", "", "Training directory to load the model parameters from (default: ./train/classifier).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("method", "hard", "Majority Vote (hard) or Weighted Average Probabilities (soft)")

FLAGS = tf.app.flags.FLAGS


def majority_vote():
    # If using majority vote, specifiy train_dir as a directory with the previously generated .txt answer files
    # Put the file you want as the default to break ties first
    print ("Using Majority Vote")
    os.chdir(FLAGS.train_dir)
    files = glob.glob('./*.txt')
    print(files)

    print ("Reading in answers")
    d = defaultdict(list)
    ensemble_answers = {}
    for fname in files:
        with open(fname) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            for line in content:
                file_name, class_name = tuple(line.split())
                d[file_name].append(class_name)

    print ("Determining Answers")
    for fname, answers in d.items():
        answer = max(answers, key = answers.count)
        ensemble_answers[fname] = answer

    print ("Writing to File")
    if not os.path.exists("majority_vote"):
        os.makedirs("ensemble")

    with open(pjoin("ensemble", 'tromero1.txt'), 'w') as f:
        for fname, answer in ensemble_answers.items():
            print(fname + " " + answer, file=f)


def weighted_average_prob():
    raise Exception("NotImplementedError")
    # TODO: Iterate through models, read them in, generate softmax probabilites for each answer, 


def main(_):
    print(vars(FLAGS))

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)
    
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ======== Answer with an Ensemble ========
    if FLAGS.method == "hard":
        majority_vote()
    elif FLAGS.method == "soft":
        weighted_average_prob()
    else:
        raise Exception("InvalidMethodError")

if __name__ == "__main__":
  tf.app.run()
