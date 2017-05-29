import io
import os
import json
import sys
import random
import re
from os.path import join as pjoin

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

# Hyperparams
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "Learning rate.") # For CS224, we used 0.001
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")    # Typically larger for cnns than rnns

# Convenience
tf.app.flags.DEFINE_string("classifier", "DemoClassifier", "The name of the classifier to use. For easily switching between classifiers.")
tf.app.flags.DEFINE_string("data_dir", "data/tiny-imagenet-200", "tiny-imagenet directory (default ./data/tiny-imagenet-200)")
tf.app.flags.DEFINE_string("train_dir", "", "Training directory to load the model parameters from (default: ./train/classifier).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("run_name", "", "A name to give the run. For checkpoint saving. Defaults to classifier name.")
tf.app.flags.DEFINE_bool("tb", False, "Log Tensorboard Graph")
tf.app.flags.DEFINE_bool("background", False, "Prettier logging if running in background")
tf.app.flags.DEFINE_bool("debug", False, "Run on a small set of data for debugging.")
tf.app.flags.DEFINE_integer("n_classes", 200, "The number of classes. Don't change.")

FLAGS = tf.app.flags.FLAGS

def process_answers(unprocessed_answers):
    """
    Extract file_name and file_number, plus reorganize into a dictionary.
    """
    answers = {}
    for file_path, wnid_prediction in unprocessed_answers:
        file_name = file_path.split("/")[-1]
        file_number = int(re.findall(r'\d+', file_name)[0])
        answers[file_number] = (file_name, wnid_prediction)
    assert(len(answers) == len(unprocessed_answers))
    return answers


def generate_answers(sess, model, dataset):
    """
    Loop over the dev or test dataset and generate answer.

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :return:
    """

    test_data = list(zip(dataset["X_test"], dataset["test_image_names"]))
    label_to_wnid = dataset["label_to_wnid"]

    batches, num_batches = get_batches(test_data, FLAGS.batch_size)

    answers = []
    for batch in tqdm(batches):
        X_batch, batch_names = zip(*batch)
        preds = model.classify(sess, X_batch)
        for i in range(len(preds)):
            answers.append((batch_names[i], label_to_wnid[preds[i]]))

    print ("Generated {}/{} Answers".format(len(answers), len(test_data)))
    return process_answers(answers)


def main(_):
    print(vars(FLAGS))

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)
    
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    print ("Loading Tiny-Imagenet Dataset")
    dataset = load_tiny_imagenet(FLAGS.data_dir, is_training = False, dtype=np.float32, subtract_mean=True, debug=FLAGS.debug)

    #Store img sizes
    FLAGS.img_H = dataset["X_test"].shape[1]
    FLAGS.img_W = dataset["X_test"].shape[2]
    FLAGS.img_C = dataset["X_test"].shape[3]
    print ("Imgs are (" + str(FLAGS.img_H) + ", " + str(FLAGS.img_W) + ", " + str(FLAGS.img_C) + ")")

    # ========= Model-specific =========
    print ("Creating '" + FLAGS.classifier + "'")
    classifier = get_classifier(FLAGS.classifier, FLAGS)

    print ("Creating Model")
    model = Model(classifier, FLAGS)

    with tf.Session() as sess:
        if FLAGS.train_dir == "":
            FLAGS.train_dir = pjoin("train", classifier.name())
        print ("train_dir: ", FLAGS.train_dir)
        initialize_model(sess, model, FLAGS.train_dir)

        print ("Generating Answers")
        answers = generate_answers(sess, model, dataset)

        # write to json file to root dir
        print ("Writing to File")
        with io.open(pjoin(FLAGS.train_dir, 'tromero1.txt'), 'w') as f:
            for _, (file_name, wnid_prediction) in sorted(answers.items()):
                print(file_name + " " + wnid_prediction, file=f)


if __name__ == "__main__":
  tf.app.run()
