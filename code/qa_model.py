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

from evaluate import exact_match_score, f1_score
from utils import beta_summaries, get_batches

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

    def image_classify(self, knowledge_rep, paragraph_mask, cell_init):
        """
        param knowledge_rep: it is a representation of the paragraph and question
        return: tuple that contains the logits for the distributions of start and end token
        """

        return tuple(preds) # Bs, Be [batchsize, paragraph_length]


class Model(object):
    def __init__(self, classifier, FLAGS, *args):
        """
        Initializes your System
        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.classifier = classifier
        self.FLAGS = FLAGS

        # ==== set up variables ========
        self.learning_rate = tf.Variable(float(self.FLAGS.learning_rate), trainable = False, name = "learning_rate")
        self.global_step = tf.Variable(int(0), trainable = False, name = "global_step")

        # # ==== set up placeholder tokens ======== 3d (because of batching)
        self.dropout_placeholder = tf.placeholder(tf.float32, (), name="dropout_placeholder")

        # ==== assemble pieces ====
        with tf.variable_scope("classifier", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_system()
            self.setup_loss()
            self.setup_predictions()

        # ==== set up training/updating procedure ==
        opt_function = get_optimizer(self.FLAGS.optimizer)  #Default is Adam
        self.decayed_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, decay_steps = 2000, decay_rate = 0.95, staircase=True)
        self.learning_rate_tb = tf.summary.scalar("learning_rate", self.decayed_rate)
        optimizer = opt_function(self.decayed_rate)

        grads_and_vars = optimizer.compute_gradients(self.loss, tf.trainable_variables())

        grads = [g for g, v in grads_and_vars]
        variables = [v for g, v in grads_and_vars]

        clipped_grads, self.global_norm = tf.clip_by_global_norm(grads, self.FLAGS.max_gradient_norm)
        self.global_norm_tb = tf.summary.scalar("global_norm", self.global_norm)
        self.train_op = optimizer.apply_gradients(zip(clipped_grads, variables), global_step = self.global_step, name = "apply_clipped_grads")

        self.saver = tf.train.Saver(tf.global_variables())


    def setup_system(self):
        # Get classification scores
        with vs.variable_scope("classify"):
            self.pred = self.classifier.image_classify(Hr, self.paragraph_mask_placeholder, self.cell_initial_placeholder)


    def setup_predictions(self):
        with vs.variable_scope("prediction"):
            self.Beta = tf.nn.softmax(self.pred)


    def setup_loss(self):
        with vs.variable_scope("loss"):
            l = tf.nn.sparse_softmax_cross_entropy_with_logits(self.pred_s, self.start_answer_placeholder)
            self.loss = tf.reduce_mean(l)

            self.train_loss_tb = tf.summary.scalar("train_loss", self.loss)
            self.val_loss_tb = tf.summary.scalar("val_loss", self.loss)


    def score(self, session, qs, ps, q_masks, p_masks):
        """
        Returns the probability distribution over different classes
        so that other methods like self.answer() will be able to work properly
        """
        input_feed = {}

        input_feed[self.question_placeholder] = np.array(list(qs))
        input_feed[self.paragraph_placeholder] = np.array(list(ps))
        input_feed[self.paragraph_mask_placeholder] = np.array(list(p_masks))
        input_feed[self.paragraph_length] = np.sum(list(p_masks), axis = 1)   # Sum and make into a list
        input_feed[self.question_length] = np.sum(list(q_masks), axis = 1)    # Sum and make into a list
        input_feed[self.cell_initial_placeholder] = np.zeros((len(qs), self.FLAGS.state_size))
        input_feed[self.dropout_placeholder] = 1

        output_feed = [self.Beta_s, self.Beta_e]    # Get the softmaxed outputs

        outputs = session.run(output_feed, input_feed)

        return outputs


    def classify(self, session, question, paragraph, question_mask, paragraph_mask):

        beta = self.score(session, question, paragraph, question_mask, paragraph_mask)
        answer = np.argmax(beta)    # ???
        return answer

    def evaluate_answer(self, session, dataset, rev_vocab, sample=100, log=False):
        """
        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        return class


    def optimize(self, session, batch):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        train_qs, train_q_masks, train_ps, train_p_masks, train_spans, train_answers = zip(*batch)    # Unzip batch, each returned element is a tuple of lists

        input_feed = {}

        start_answers = [train_span[0] for train_span in list(train_spans)]
        end_answers = [train_span[1] for train_span in list(train_spans)]

        input_feed[self.question_placeholder] = np.array(list(train_qs))
        input_feed[self.paragraph_placeholder] = np.array(list(train_ps))
        input_feed[self.start_answer_placeholder] = np.array(start_answers)
        input_feed[self.end_answer_placeholder] = np.array(end_answers)
        input_feed[self.paragraph_mask_placeholder] = np.array(list(train_p_masks))
        input_feed[self.paragraph_length] = np.sum(list(train_p_masks), axis = 1)   # Sum and make into a list
        input_feed[self.question_length] = np.sum(list(train_q_masks), axis = 1)    # Sum and make into a list
        input_feed[self.dropout_placeholder] = self.FLAGS.dropout
        input_feed[self.cell_initial_placeholder] = np.zeros((len(train_qs), self.FLAGS.state_size))

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


    def train(self, session, dataset, train_dir):
        """
        Implement main training loop
        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.
        More ambitious approach can include implement early stopping, or reload
        previous models if they have higher performance than the current one
        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.
        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.
        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param train_dir: path to the directory where you should save the model checkpoint
        :return:
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

        checkpoint_path = os.path.join(train_dir, rname)
        early_stopping_path = os.path.join(checkpoint_path, "early_stopping")

        train_data = zip(dataset["train_questions"], dataset["train_questions_mask"], dataset["train_context"], dataset["train_context_mask"], dataset["train_span"], dataset["train_answer"])
        val_data = zip(dataset["val_questions"], dataset["val_questions_mask"], dataset["val_context"], dataset["val_context_mask"], dataset["val_span"], dataset["val_answer"])

        num_data = len(train_data)
        best_f1 = 0

        # Normal training loop
        rolling_ave_window = 50
        losses = [10]*rolling_ave_window

        for cur_epoch in range(self.FLAGS.epochs):
            batches, num_batches = get_batches(train_data, self.FLAGS.batch_size)

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

            #Save model after each epoch
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            save_path = saver.save(session, os.path.join(checkpoint_path, "model.ckpt"), step)
            logging.info("Model checkpoint saved in file: %s" % save_path)

            logging.info("---------- Evaluating on Train Set ----------")
            self.evaluate_answer(session, train_data, rev_vocab, sample=self.FLAGS.eval_size, log=True)
            logging.info("---------- Evaluating on Val Set ------------")
            f1, em = self.evaluate_answer(session, val_data, rev_vocab, sample=self.FLAGS.eval_size, log=True)

            # Save best model based on F1 (Early Stopping)
            if f1 > best_f1:
                best_f1 = f1
                if not os.path.exists(early_stopping_path):
                    os.makedirs(early_stopping_path)
                save_path = saver.save(session, os.path.join(early_stopping_path, "best_model.ckpt"))
                logging.info("New Best F1 Score: %f !!! Best Model saved in file: %s" % (best_f1, save_path))


