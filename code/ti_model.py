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
from data_utils import augment_batch, crop_10
from lrmanager import lrManager

logging.basicConfig(level=logging.INFO)


class Model(object):
    def __init__(self, classifier, FLAGS, *args):
        """
        Initializes your System
        :param classifier: an image classifier that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.classifier = classifier
        self.FLAGS = FLAGS
        self.current_lr =  self.FLAGS.learning_rate

        # ==== set up variables ========
        # self.learning_rate = tf.Variable(float(self.FLAGS.learning_rate), trainable = False, name = "learning_rate")
        self.global_step = tf.Variable(int(0), trainable = False, name = "global_step")

        # # ==== set up placeholder tokens ======== 3d (because of batching)
        self.learning_rate = tf.placeholder(tf.float32, name = "learning_rate")
        self.X = tf.placeholder(tf.float32, [None, FLAGS.img_H, FLAGS.img_W, FLAGS.img_C], name="X")
        self.y = tf.placeholder(tf.int64, [None], name="y")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

        # ==== assemble pieces ====
        with tf.variable_scope("model"):
            self.setup_system()
            self.setup_loss()
            self.setup_training_procedure()

        # ==== setup saver ====
        self.saver = tf.train.Saver(tf.global_variables())


    def setup_system(self):
        with vs.variable_scope("classify"):
            raw_scores = self.classifier.forward_pass(self.X, self.is_training)
            self.y_out = tf.nn.softmax(raw_scores, name = "softmax")    # Apply softmax to raw output scores

            with tf.name_scope('y_out_summaries'):
                mean = tf.reduce_mean(self.y_out)
                stddev = tf.sqrt(tf.reduce_mean(tf.square(self.y_out - mean)))
                self.y_out_stddev_tb = tf.summary.scalar('stddev', stddev)
                self.y_out_max_tb = tf.summary.scalar('max', tf.reduce_max(self.y_out))


    def setup_loss(self):
        with vs.variable_scope("loss"):
            self.loss = self.classifier.loss(self.y)

            self.train_loss_tb = tf.summary.scalar("train_loss", self.loss)
            self.val_loss_tb = tf.summary.scalar("val_loss", self.loss)


    def setup_training_procedure(self):
        with vs.variable_scope("train_op"):
            self.train_op, self.global_norm = self.classifier.train_op(self.learning_rate, self.global_step, self.loss)

            self.learning_rate_tb = tf.summary.scalar("learning_rate", self.learning_rate)
            self.global_norm_tb = tf.summary.scalar("global_norm", self.global_norm)
        

    def score(self, session, X_batch):
        """
        NOT FOR TRAINING
        
        Returns the scores over different classes so that other methods
        like self.answer() will be able to work properly       
        """
        input_feed = {}

        input_feed[self.X] = X_batch
        input_feed[self.is_training] = False

        output_feed = [self.y_out]

        outputs = session.run(output_feed, input_feed)
        outputs = outputs[0]    # Run returns the outputfeed as a list. We just want the first element

        return np.array(outputs)


    def crop_classify(self, session, image, top5 = False):
        '''
        NOT FOR TRAINING

        Returns: predicted class identifier after
            averageing softmax results from each
            image in the collection.
        
        Intended for classifying crops of the same image
        '''
        assert(image.shape[0] == 1)

        if(self.FLAGS.augment):
            crops = crop_10(image, self.FLAGS.img_H, self.FLAGS.img_W)
        else:
            crops = image

        scores = self.score(session, crops)
        overall_score = np.mean(scores, axis=0)

        if not top5:
            pred = np.argmax(overall_score)
            return pred
        else:
            top5pred = np.argpartition(overall_score, -5)[-5:]
            return top5pred


    def classify(self, session, X_batch):
        '''
        NOT FOR TRAINING

        Returns: predicted class identifier
        '''

        scores = self.score(session, X_batch)
        preds = np.argmax(scores, axis=1)
        return preds


    def evaluate_model(self, session, dataset, sample_size=None, top5 = False):
        """
        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return: prediction accuracy
        """
        if sample_size == None:
            sample_size = len(dataset)

        eval_set = random.sample(dataset, sample_size)

        running_sum = 0
        for img, label in eval_set:
            img = np.expand_dims(img, axis=0)
            pred = self.crop_classify(session, img, top5)
            correct_pred = np.equal(pred, label)
            running_sum += np.sum(correct_pred)

        accuracy = running_sum/sample_size
        return accuracy


    def optimize(self, session, batch):
        """
        FOR TRAINING ONLY
        
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
            loss, global_norm, global_step
        """
        X_batch, y_batch = zip(*batch)    # Unzip batch, each returned element is a tuple of lists

        if(self.FLAGS.augment):
            X_batch = augment_batch(X_batch, self.FLAGS.img_H, self.FLAGS.img_W)

        input_feed = {}

        input_feed[self.X] = X_batch
        input_feed[self.y] = y_batch
        input_feed[self.is_training] = True
        input_feed[self.learning_rate] = self.current_lr

        output_feed = []

        output_feed.append(self.train_op)
        output_feed.append(self.loss)
        output_feed.append(self.global_norm)
        output_feed.append(self.global_step)


        if self.FLAGS.tb is True:
            output_feed.append(self.train_loss_tb)
            output_feed.append(self.global_norm_tb)
            output_feed.append(self.learning_rate_tb)
            output_feed.append(self.y_out_stddev_tb)
            output_feed.append(self.y_out_max_tb)
            tr, loss, norm, step, train_tb, norm_tb, lr_tb, y_out_stddev_tb, y_out_max_tb = session.run(output_feed, input_feed)
            self.tensorboard_writer.add_summary(train_tb, step)
            self.tensorboard_writer.add_summary(norm_tb, step)
            self.tensorboard_writer.add_summary(lr_tb, step)
            self.tensorboard_writer.add_summary(y_out_stddev_tb, step)
            self.tensorboard_writer.add_summary(y_out_max_tb, step)
        else:
            tr, loss, norm, step = session.run(output_feed, input_feed)

        return loss, norm, step


    def train(self, session, dataset):
        """
        Main training loop

        Tips:
        look into tf.train.exponential_decay)
        You should save your model per epoch.
        Implement early stopping
        Evaluate your training progress by printing out information
        We recommend you evaluate your model performance on accuracy instead of just loss

        :param session: it should be passed in from train.py
        :param dataset: A dictionary with the following entries:
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
        # Setup Tensorboard
        if self.FLAGS.tb is True:
            tensorboard_path = os.path.join(self.FLAGS.log_dir, "tensorboard")
            self.tensorboard_writer = tf.summary.FileWriter(tensorboard_path, session.graph)

        #Info for saving models
        saver = tf.train.Saver()
        if self.FLAGS.run_name is "": rname = self.classifier.name()
        else: rname = self.FLAGS.run_name
        checkpoint_path = os.path.join(self.FLAGS.train_dir, rname)
        early_stopping_path = os.path.join(checkpoint_path, "early_stopping")

        # Setup conveinient handles on train and val sets
        train_data = list(zip(dataset["X_train"], dataset["y_train"]))
        val_data = list(zip(dataset["X_val"], dataset["y_val"]))

         # Helper stuff
        num_data = len(train_data)
        best_val_acc = 0
        best_train_acc = 0
        rolling_ave_window = 10
        losses = [10]*rolling_ave_window

        # Systematic learning rate decay
        lrHelper = lrManager(self.FLAGS, num_data)
        
        # Epoch level loop
        step = 1
        for cur_epoch in range(self.FLAGS.epochs):
            batches, num_batches = get_batches(train_data, self.FLAGS.batch_size)
            
            # Training loop
            for _i, batch in enumerate(batches):
                i = _i + 1  # For convienince

                self.current_lr = lrHelper.get_lr(step)

                #Optimatize using batch
                loss, norm, step = self.optimize(session, batch)
                losses[step % rolling_ave_window] = loss
                mean_loss = np.mean(losses)

                #Print relevant params
                num_complete = int(20*(self.FLAGS.batch_size*i/num_data))
                if self.FLAGS.background:
                    logging.info("EPOCH: %d ==> (Avg Loss: %.3f <-> Batch Loss: %.3f) [%-20s] (%d/%d) [norm: %.2f] [lr: %f] [step: %d]"
                        % (cur_epoch + 1, mean_loss, loss, '='*num_complete, min(i*self.FLAGS.batch_size, num_data), num_data, norm, self.current_lr, step))
                else:
                    sys.stdout.write('\r')
                    sys.stdout.write("EPOCH: %d ==> (Avg Loss: %.3f <-> Batch Loss: %.3f) [%-20s] (%d/%d) [norm: %.2f] [lr: %f] [step: %d]"
                        % (cur_epoch + 1, mean_loss, loss, '='*num_complete, min(i*self.FLAGS.batch_size, num_data), num_data, norm, self.current_lr, step))
                    sys.stdout.flush()

                # Save model snapshots
                if lrHelper.save_snapshot(step):
                    snapshot_path = os.path.join(checkpoint_path, "snap" + str(lrHelper.snapshot_num(step)))
                    if not os.path.exists(snapshot_path):
                        os.makedirs(snapshot_path)
                    save_path = saver.save(session, os.path.join(snapshot_path, "model.ckpt"))
                    logging.info("\nSnapshot saved at:  %s \n" % (save_path))

            sys.stdout.write('\n')

            # Evaluate accuracy
            eval_size = min(len(val_data), len(train_data))//2
            train_acc = self.evaluate_model(session, train_data, eval_size)
            logging.info("Training Accuracy: %f \t\ton %d examples" % (train_acc, eval_size))
            val_acc = self.evaluate_model(session, val_data, eval_size)
            logging.info("Validation Accuracy: %f \ton %d examples" % (val_acc, eval_size))

            # Save best model based on accuracy (Early Stopping)
            if val_acc > best_val_acc and self.FLAGS.debug == False:
                best_val_acc = val_acc
                if not os.path.exists(checkpoint_path):
                    os.makedirs(checkpoint_path)
                save_path = saver.save(session, os.path.join(checkpoint_path, "model.ckpt"))
                logging.info("New Best Validation Accuracy: %f !!! Best Model saved in file: %s" % (best_val_acc, save_path))

            


                
