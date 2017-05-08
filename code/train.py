import os
import json

import tensorflow as tf

from ti_model import Model
from ti_classifiers import get_classifier
from utils import *
from data_utils import *

from os.path import join as pjoin
import logging

logging.basicConfig(level=logging.INFO)

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "Learning rate.") # For CS224, we used 0.001
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.5, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size to use during training.")    # Typically larger for cnns than rnns

# Training Settings
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("epochs", 15, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("eval_size", 2560, "The number of examples to evaluate on.")

# Convenience
tf.app.flags.DEFINE_string("classifier", "DemoClassifier", "The name of the classifier to use. For easily switching between classifiers.")
tf.app.flags.DEFINE_string("data_dir", "data/tiny-imagenet-200", "tiny-imagenet directory (default ./data/tiny-imagenet-200)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("run_name", "", "A name to give the run. For checkpoint saving. Defaults to classifier name.")
tf.app.flags.DEFINE_bool("tb", False, "Log Tensorboard Graph")
tf.app.flags.DEFINE_bool("background", False, "Prettier logging if running in background")
tf.app.flags.DEFINE_bool("debug", False, "Run on a small set of data for debugging.")
tf.app.flags.DEFINE_integer("n_classes", 200, "The number of classes. Don't change.")

FLAGS = tf.app.flags.FLAGS

def main(_):

    ### Load tiny-imagenet-200 ###
    """
    Returns a dictionary with the following entries:
    - class_names: A list where class_names[i] is a list of strings giving the
      WordNet names for class i in the loaded dataset.
    - X_train: (N_tr, 64, 64, 3) array of training images
    - y_train: (N_tr,) array of training labels
    - X_val: (N_val, 64, 64, 3) array of validation images
    - y_val: (N_val,) array of validation labels
    - X_test: (N_test, 64, 64, 3) array of testing images.
    - y_test: (N_test,) array of test labels; if test labels are not available
      (such as in student code) then y_test will be None.
    - mean_image: (64, 64, 3) array giving mean training image
    """
    print(vars(FLAGS))

    print ("Loading Tiny-Imagenet Dataset")
    dataset = load_tiny_imagenet(FLAGS.data_dir, is_training = True, dtype=np.float32, subtract_mean=True, debug=FLAGS.debug)

    print ("Number of Classes: ", len(dataset["class_names"]))
    FLAGS.n_classes = len(dataset["class_names"])

    print ("Creating '" + FLAGS.classifier + "'")
    classifier = get_classifier(FLAGS.classifier, FLAGS)

    print ("Creating Model")
    model = Model(classifier, FLAGS)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    print ("Starting TF Session")
    with tf.Session() as sess:
        load_train_dir = FLAGS.load_train_dir or FLAGS.train_dir
        print ("load_train_dir: ", load_train_dir)

        print ("Initializing Model")
        initialize_model(sess, model, load_train_dir)

        print ("\n")
        model.train(sess, dataset)

if __name__ == "__main__":
    tf.app.run()
