from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import copy
import random
import sys
import math
from datetime import datetime

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.nn import sparse_softmax_cross_entropy_with_logits

from utils import get_batches

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class ImageClassifier(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

    def image_classify(self, X, is_training):
        """
        NOTE: Data is in the format NCHW, not NHWC

        param X: A batch of image data
        param is_training: Whether or not this is a training or testing batch
        return: tuple that contains the logits for the distributions of start and end token
        """
        # Just my model from CS231n Assignment 2
        # Conv Layers
        conv1 = tf.contrib.layers.conv2d(X, num_outputs=64, kernel_size=3, stride=1, data_format='NCHW', padding='VALID', scope = "Conv1")
        bn1 = tf.contrib.layers.batch_norm(conv1, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn1")
        mp1 = tf.nn.max_pool(bn1, [1,2,2,1], strides=[1,2,2,1], padding='VALID', data_format='NCHW', name="max_pool1")
        
        conv2 = tf.contrib.layers.conv2d(mp1, num_outputs=64, kernel_size=4, stride=1, data_format='NCHW', padding='VALID', scope = "Conv2")
        bn2 = tf.contrib.layers.batch_norm(conv2, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn2")
        mp2 = tf.nn.max_pool(bn2, [1,2,2,1], strides=[1,2,2,1], padding='VALID', data_format='NCHW', name="max_pool2")
        
        conv3 = tf.contrib.layers.conv2d(mp2, num_outputs=32, kernel_size=5, stride=1, data_format='NCHW', padding='VALID', scope = "Conv3")
        bn3 = tf.contrib.layers.batch_norm(conv3, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn3")
        mp3 = tf.nn.max_pool(bn3, [1,2,2,1], strides=[1,2,2,1], padding='VALID', data_format='NCHW', name="max_pool3")
        
        # Affine Layers
        h1_flat = tf.contrib.layers.flatten(mp3)
        fc1 = tf.contrib.layers.fully_connected(inputs = h1_flat, num_outputs = 512, scope = "fc1")
        raw_scores = tf.contrib.layers.fully_connected(inputs = fc1, num_outputs = self.FLAGS.n_classes, activation_fn = None, scope = "fc2")

        return raw_scores


class Model(object):
    def __init__(self, classifier, FLAGS, *args):
        """
        Initializes your System
        :param classifier: an image classifier that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.classifier = classifier
        self.FLAGS = FLAGS

        # ==== set up variables ========
        self.learning_rate = tf.Variable(float(self.FLAGS.learning_rate), trainable = False, name = "learning_rate")
        self.global_step = tf.Variable(int(0), trainable = False, name = "global_step")

        # # ==== set up placeholder tokens ======== 3d (because of batching)
        #self.dropout_placeholder = tf.placeholder(tf.float32, (), name="dropout_placeholder")
        self.X = tf.placeholder(tf.float32, [None, 3, 64, 64], name="X")
        self.y = tf.placeholder(tf.int64, [None], name="y")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        # ==== assemble pieces ====
        with tf.variable_scope("classifier", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_system()
            self.setup_loss()
            self.setup_training_procedure()


    def setup_training_procedure(self):
        opt_function = get_optimizer(self.FLAGS.optimizer)  #Default is Adam
        self.decayed_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps = 2000, decay_rate = 0.95, staircase=True)
        optimizer = opt_function(self.decayed_rate)

        grads_and_vars = optimizer.compute_gradients(self.loss, tf.trainable_variables())
        grads = [g for g, v in grads_and_vars]
        variables = [v for g, v in grads_and_vars]

        clipped_grads, self.global_norm = tf.clip_by_global_norm(grads, self.FLAGS.max_gradient_norm)

        # Batch Norm in tensorflow requires this extra dependency
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.apply_gradients(zip(clipped_grads, variables), global_step = self.global_step, name = "apply_clipped_grads")

        self.learning_rate_tb = tf.summary.scalar("learning_rate", self.decayed_rate)
        self.global_norm_tb = tf.summary.scalar("global_norm", self.global_norm)
        self.saver = tf.train.Saver(tf.global_variables())


    def setup_system(self):
        # Get classification scores
        with vs.variable_scope("classify"):
            self.raw_scores = self.classifier.image_classify(self.X, self.is_training)
        with vs.variable_scope("predict"):
            self.y_out = tf.nn.softmax(self.raw_scores)


    def setup_loss(self):
        with vs.variable_scope("loss"):
            l = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.y, self.FLAGS.n_classes),logits=self.y_out)
            self.loss = tf.reduce_mean(l)

            self.train_loss_tb = tf.summary.scalar("train_loss", self.loss)
            self.val_loss_tb = tf.summary.scalar("val_loss", self.loss)


    def score(self, session, X_batch):
        """
        Returns the probability distribution over different classes
        so that other methods like self.answer() will be able to work properly

        NOT FOR TRAINING
        """
        input_feed = {}

        input_feed[self.X] = X_batch
        input_feed[self.is_training] = False
        #input_feed[self.dropout_placeholder] = 1

        output_feed = [self.y_out]    # Get the softmaxed outputs

        outputs = session.run(output_feed, input_feed)

        return outputs


    def classify(self, session, X_batch):
        # Returns: predicted class identifier
        # NOT FOR TRAINING
        scores = self.score(session, X_batch)
        preds = np.argmax(scores, axis=1)
        return preds

    def evaluate_model(self, session, dataset, sample=100, log=False):
        """
        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
            prediction accuracy
        """
        eval_set = random.sample(dataset, sample)
        batches, num_batches = get_batches(eval_set, self.FLAGS.batch_size)

        running_sum = 0
        for batch in batches:
            X_batch, y_batch = zip(*batch)
            preds = self.classify(session, X_batch)
            correct_preds = tf.equal(preds, y_batch)
            running_sum += tf.reduce_sum(tf.cast(correct_preds, tf.float32))

        accuracy = running_sum/float(len(eval_set))
        return accuracy


    def optimize(self, session, batch):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
            loss, global_norm, global_step

        FOR TRAINING ONLY
        """
        X_batch, y_batch = zip(*batch)    # Unzip batch, each returned element is a tuple of lists

        input_feed = {}

        input_feed[self.X] = X_batch
        input_feed[self.y] = y_batch
        input_feed[self.is_training] = True
        #input_feed[self.dropout_placeholder] = self.FLAGS.dropout

        output_feed = []

        output_feed.append(self.train_op)
        output_feed.append(self.loss)
        output_feed.append(self.global_norm)
        output_feed.append(self.global_step)


        if self.FLAGS.tb is True:
            output_feed.append(self.train_loss_tb)
            output_feed.append(self.global_norm_tb)
            output_feed.append(self.learning_rate_tb)
            tr, loss, norm, step, train_tb, norm_tb, lr_tb = session.run(output_feed, input_feed)
            self.tensorboard_writer.add_summary(train_tb, step)
            self.tensorboard_writer.add_summary(norm_tb, step)
            self.tensorboard_writer.add_summary(lr_tb, step)
        else:
            tr, loss, norm, step = session.run(output_feed, input_feed)

        return loss, norm, step


    def train(self, session, dataset):
        """
        Implement main training loop
        TIPS:
        look into tf.train.exponential_decay)
        You should save your model per epoch.
        Implement early stopping
        Evaluate your training progress by printing out information

        We recommend you evaluate your model performance on accuracy instead of just loss

        :param session: it should be passed in from train.py
        :param dataset: A dictionary with the following entries:
                        - class_names: A list where class_names[i] is a list of strings giving the
                        WordNet names for class i in the loaded dataset.
                        - X_train: (N_tr, 3, 64, 64) array of training images
                        - y_train: (N_tr,) array of training labels
                        - X_val: (N_val, 3, 64, 64) array of validation images
                        - y_val: (N_val,) array of validation labels
                        - X_test: (N_test, 3, 64, 64) array of testing images.
                        - y_test: (N_test,) array of test labels; if test labels are not available
                        (such as in student code) then y_test will be None.
                        - mean_image: (3, 64, 64) array giving mean training image
        """
        if self.FLAGS.tb is True:
            tensorboard_path = os.path.join(self.FLAGS.log_dir, "tensorboard")
            self.tensorboard_writer = tf.summary.FileWriter(tensorboard_path, session.graph)

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))

        #Info for saving models
        saver = tf.train.Saver()
        if self.FLAGS.run_name == "":
            rname = "{:%d-%m-%Y_%H:%M:%S}".format(datetime.now())
        else:
            rname = self.FLAGS.run_name

        checkpoint_path = os.path.join(self.FLAGS.train_dir, rname)
        early_stopping_path = os.path.join(checkpoint_path, "early_stopping")

        train_data = zip(dataset["X_train"], dataset["y_train"])
        val_data = zip(dataset["X_val"], dataset["y_val"])

        num_data = len(train_data)
        best_acc = 0
        rolling_ave_window = 50
        losses = [10]*rolling_ave_window

        # Epoch level loop
        for cur_epoch in range(self.FLAGS.epochs):
            batches, num_batches = get_batches(train_data, self.FLAGS.batch_size)

            # Training loop
            for i, batch in enumerate(batches):
                #Optimatize using batch
                loss, norm, step = self.optimize(session, batch)
                losses[step % rolling_ave_window] = loss
                mean_loss = np.mean(losses)

                #Print relevant params
                num_complete = int(20*(self.FLAGS.batch_size*float(i+1)/num_data))
                if not self.FLAGS.background:
                    sys.stdout.write('\r')
                    sys.stdout.write("EPOCH: %d ==> (Avg Loss: [Train: %.3f][Val: %.3f] <--> Batch Loss: %.3f) [%-20s] (Completion:%d/%d) [norm: %.2f] [Step: %d]" % (cur_epoch + 1, mean_loss, mean_val_loss, loss, '='*num_complete, (i+1)*self.FLAGS.batch_size, num_data, norm, step))
                    sys.stdout.flush()
                else:
                    logging.info("EPOCH: %d ==> (Avg Loss: [Train: %.3f][Val: %.3f] <--> Batch Loss: %.3f) [%-20s] (Completion:%d/%d) [norm: %.2f] [Step: %d]" % (cur_epoch + 1, mean_loss, mean_val_loss, loss, '='*num_complete, (i+1)*self.FLAGS.batch_size, num_data, norm, step))

            sys.stdout.write('\n')

            #Save model after each epoch. Do we really want to do this? Maybe just save the best one?
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            save_path = saver.save(session, os.path.join(checkpoint_path, "model.ckpt"), step)
            logging.info("Model checkpoint saved in file: %s" % save_path)

            logging.info("---------- Evaluating on Train Set ----------")
            self.evaluate_model(session, train_data, sample=self.FLAGS.eval_size, log=True)
            logging.info("---------- Evaluating on Val Set ------------")
            f1, em = self.evaluate_model(session, val_data, sample=self.FLAGS.eval_size, log=True)

            # Save best model based on F1 (Early Stopping)
            if acc > best_acc:
                best_acc = acc
                if not os.path.exists(early_stopping_path):
                    os.makedirs(early_stopping_path)
                save_path = saver.save(session, os.path.join(early_stopping_path, "best_model.ckpt"))
                logging.info("New Best Validation Accuracy: %f !!! Best Model saved in file: %s" % (best_acc, save_path))


