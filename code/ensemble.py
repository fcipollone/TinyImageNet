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

tf.app.flags.DEFINE_float("learning_rate", 0.0005, "Learning rate.") # For CS224, we used 0.001
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 256, "Batch size to use during training.")    # Typically larger for cnns than rnns
tf.app.flags.DEFINE_string("optimizer", "adam", "The name of the classifier to use. For easily switching between classifiers.")
tf.app.flags.DEFINE_float("weight_decay", 0.0001, "Weight decay coefficient, some models may not use this")

# Convenience
tf.app.flags.DEFINE_string("classifier", "DemoClassifier", "The name of the classifier to use. For easily switching between classifiers.")
tf.app.flags.DEFINE_string("train_dir", "", "Training directory to load the model parameters from (default: ./train/classifier).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("data_dir", "data/tiny-imagenet-200", "tiny-imagenet directory (default ./data/tiny-imagenet-200)")
tf.app.flags.DEFINE_bool("debug", False, "Run on a small set of data for debugging.")
tf.app.flags.DEFINE_bool("augment", True, "Whether or not to expand dataset using data augmentation")
tf.app.flags.DEFINE_integer("n_classes", 200, "The number of classes. Don't change.")

tf.app.flags.DEFINE_string("method", "soft", "Majority Vote (hard) or Average Probabilities (soft)")

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
    if not os.path.exists("mv_ensemble"):
        os.makedirs("mv_ensemble")

    with open(pjoin("mv_ensemble", 'tromero1.txt'), 'w') as f:
        for fname, answer in ensemble_answers.items():
            print(fname + " " + answer, file=f)


def average_prob():
    print ("Loading Tiny-Imagenet Dataset")
    dataset = load_tiny_imagenet(FLAGS.data_dir, is_training = False, dtype=np.float32, subtract_mean=True, debug=FLAGS.debug)
    label_to_wnid = dataset["label_to_wnid"]

    #Store img sizes
    jitter = 8
    FLAGS.img_H = dataset["X_test"].shape[1] - jitter
    FLAGS.img_W = dataset["X_test"].shape[2] - jitter
    FLAGS.img_C = dataset["X_test"].shape[3]
    print ("Imgs are (" + str(FLAGS.img_H) + ", " + str(FLAGS.img_W) + ", " + str(FLAGS.img_C) + ")")

    # ========= Model-specific =========
    os.chdir(FLAGS.train_dir)
    folders = glob.glob("./*/")
    print(folders)

    print ("Creating '" + FLAGS.classifier + "'")
    classifier = get_classifier(FLAGS.classifier, FLAGS)

    print ("Creating Model")
    model = Model(classifier, FLAGS)

    # Generate scores for each model
    all_scores = []
    for i, folder in enumerate(folders):
        with tf.Session() as sess:
            print ("model dir: ", folder)
            initialize_model(sess, model, folder)

            print ("Generating Scores")
            scores = generate_scores(sess, model, dataset)
            all_scores.append(scores)

    # Average of scores for each model
    answers = {}
    for img_file in all_scores[0]:
        scores = [scores_dict[img_file] for scores_dict in all_scores]
        scores = np.mean(np.stack(scores), axis = 0)
        pred = np.argmax(scores)
        answers[img_file] = label_to_wnid[pred]

    answers = process_answers(answers)

    # write to json file to root dir
    print ("Writing to File")
    with io.open(pjoin(FLAGS.train_dir, 'tromero1.txt'), 'w') as f:
        for _, (file_name, wnid_prediction) in sorted(answers.items()):
            print(file_name + " " + wnid_prediction, file=f)


def process_answers(unprocessed_answers):
    """
    Extract file_name and file_number, plus reorganize into a dictionary.
    """
    answers = {}
    for file_name in unprocessed_answers:
        file_number = int(re.findall(r'\d+', file_name)[0])
        answers[file_number] = (file_name, unprocessed_answers[file_name])
    assert(len(answers) == len(unprocessed_answers))
    return answers


def generate_scores(sess, model, dataset):
    test_data = list(zip(dataset["X_test"], dataset["test_image_names"]))

    scores = {}
    for img, img_name in tqdm(test_data):
        file_name = img_name.split("/")[-1]
        img = np.expand_dims(img, axis=0)
        score = model.crop_classify(sess, img, raw_score = True)
        scores[file_name] = score

    print ("Generated {}/{} Scores".format(len(scores), len(test_data)))
    return scores


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
        average_prob()
    else:
        raise Exception("InvalidMethodError")

if __name__ == "__main__":
  tf.app.run()
