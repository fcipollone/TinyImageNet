from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from utils import *

from os.path import join as pjoin
import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.6, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 30, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_string("run_name", "", "A name to give the run. For checkpoint saving. Defaults to the current date and time")
tf.app.flags.DEFINE_bool("tb", False, "Log Tensorboard Graph")
tf.app.flags.DEFINE_bool("background", False, "Prettier logging if running in background")

FLAGS = tf.app.flags.FLAGS

def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    dataset = get_dataset(FLAGS.data_dir, FLAGS.max_question_size, FLAGS.max_paragraph_size)

    qa = QASystem(encoder, decoder, FLAGS)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = FLAGS.load_train_dir or FLAGS.train_dir
        print ("load_train_dir: ", load_train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = FLAGS.train_dir
        print ("save_train_dir: ", save_train_dir)
        qa.train(sess, dataset, save_train_dir, rev_vocab)

if __name__ == "__main__":
    tf.app.run()
