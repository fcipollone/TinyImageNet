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
tf.app.flags.DEFINE_integer("batch_size", 256, "Batch size to use during training.")    # Typically larger for cnns than rnns
tf.app.flags.DEFINE_string("optimizer", "adam", "The name of the classifier to use. For easily switching between classifiers.")
tf.app.flags.DEFINE_float("weight_decay", 0.0001, "Weight decay coefficient, some models may not use this")

# Convenience
tf.app.flags.DEFINE_string("classifier", "DemoClassifier", "The name of the classifier to use. For easily switching between classifiers.")
tf.app.flags.DEFINE_string("data_dir", "data/tiny-imagenet-200", "tiny-imagenet directory (default ./data/tiny-imagenet-200)")
tf.app.flags.DEFINE_string("train_dir", "", "Training directory to load the model parameters from (default: ./train/classifier).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_bool("debug", False, "Run on a small set of data for debugging.")
tf.app.flags.DEFINE_bool("augment", True, "Whether or not to expand dataset using data augmentation")
tf.app.flags.DEFINE_integer("n_classes", 200, "The number of classes. Don't change.")

FLAGS = tf.app.flags.FLAGS

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
    for img, img_name in tqdm(test_data):
        img = np.expand_dims(img, axis=0)
        pred = model.crop_classify(sess, img)
        answers.append((img_name, label_to_wnid[pred]))

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
    dataset = load_tiny_imagenet(FLAGS.data_dir, is_training = True, dtype=np.float32, subtract_mean=True, debug=FLAGS.debug)   # Get the validation set

    #Store img sizes
    FLAGS.img_C = dataset["X_train"].shape[3]
    if FLAGS.augment:
        jitter = 8
        FLAGS.img_H = dataset["X_train"].shape[1] - jitter
        FLAGS.img_W = dataset["X_train"].shape[2] - jitter
    else: 
        FLAGS.img_H = dataset["X_train"].shape[1]
        FLAGS.img_W = dataset["X_train"].shape[2]
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

        val_data = list(zip(dataset["X_val"], dataset["y_val"]))
        eval_size = len(val_data)

        top1acc = model.evaluate_model(sess, val_data, eval_size, top5 = False)
        print("Top-1 Validation Accuracy: %f \ton %d examples" % (val_acc, eval_size))

        top5acc = model.evaluate_model(sess, val_data, eval_size, top5 = True)
        print("Top-5 Validation Accuracy: %f \ton %d examples" % (val_acc, eval_size))



if __name__ == "__main__":
  tf.app.run()
