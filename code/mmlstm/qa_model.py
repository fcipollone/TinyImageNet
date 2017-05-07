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
from tensorflow.python.ops.nn import bidirectional_dynamic_rnn
from tensorflow.python.ops.nn import dynamic_rnn

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

class MatchLSTMCell(tf.nn.rnn_cell.BasicLSTMCell):
    """
    Extension of LSTM cell to do matching and magic. Designed to be fed to dynammic_rnn
    """
    def __init__(self, hidden_size, HQ, FLAGS, params, reverse = False):
         # Uniform distribution, as opposed to xavier, which is normal
        self.HQ = HQ
        self.hidden_size = hidden_size
        self.FLAGS = FLAGS
        self.reverse = reverse

        l, P, Q = self.hidden_size, self.FLAGS.max_paragraph_size, self.FLAGS.max_question_size

        if (reverse):
            P = self.FLAGS.max_question_size
            Q = self.FLAGS.max_paragraph_size

        self.WQ, self.WP, self.WR, self.bP, self.w, self.b = params

        # self.WQ = tf.get_variable("WQ", [l,l], initializer=tf.uniform_unit_scaling_initializer(1.0))
        # self.WP = tf.get_variable("WP", [l,l], initializer=tf.uniform_unit_scaling_initializer(1.0))
        # self.WR = tf.get_variable("WR", [l,l], initializer=tf.uniform_unit_scaling_initializer(1.0))

        # self.bP = tf.Variable(tf.zeros([1, l]))
        # self.w = tf.Variable(tf.zeros([l,1]))
        # self.b = tf.Variable(tf.zeros([1,1]))

        # Calculate term1 by resphapeing to l
        HQ_shaped = tf.reshape(HQ, [-1, l])
        term1 = tf.matmul(HQ_shaped, self.WQ)
        term1 = tf.reshape(term1, [-1, Q, l])
        self.term1 = term1

        super(MatchLSTMCell, self).__init__(hidden_size)

    def __call__(self, inputs, state, scope = None):
        """
        inputs: a batch representation (HP at each word i) that is inputs = hp_i and are [None, l]
        state: a current state for our cell which is LSTM so its a tuple of (c_mem, h_state), both are [None, l]
        """

        #For naming convention load in from self the params and rename
        term1 = self.term1
        WQ, WP, WR = self.WQ, self.WP, self.WR
        bP, w, b = self.bP, self.w, self.b
        l, P, Q = self.hidden_size, self.FLAGS.max_paragraph_size, self.FLAGS.max_question_size
        if (self.reverse):
            P = self.FLAGS.max_question_size
            Q = self.FLAGS.max_paragraph_size

        HQ = self.HQ
        hr = state[1]
        hp_i = inputs

        # Check correct input dimensions
        assert hr.get_shape().as_list() == [None, l]
        assert hp_i.get_shape().as_list() == [None, l]

        # Way to extent a [None, l] matrix by dim Q
        term2 = tf.matmul(hp_i,WP) + tf.matmul(hr, WR) + bP
        term2 = tf.tile(tf.expand_dims(term2,1),[1,Q,1])
        #term2 = tf.transpose(tf.stack([term2 for _ in range(Q)]), [1,0,2])

        # Check correct term dimensions for use
        assert term1.get_shape().as_list() == [None, Q, l]
        assert term2.get_shape().as_list() == [None, Q, l]

        # Yeah pretty sure we need this lol
        G_i = tf.tanh(term1 + term2)

        # Reshape to multiply against w
        G_i_shaped = tf.reshape(G_i, [-1, l])
        a_i = tf.matmul(G_i_shaped, w) + b
        a_i = tf.reshape(a_i, [-1, Q, 1])

        # Check that the attention matrix is properly shaped (3rd dim useful for batch_matmul in next step)
        assert a_i.get_shape().as_list() == [None, Q, 1]

        # Prepare dims, and mult attn with question representation in each element of the batch
        HQ_shaped = tf.transpose(HQ, [0,2,1])
        z_comp = tf.batch_matmul(HQ_shaped, a_i)
        z_comp = tf.squeeze(z_comp, [2])

        # Check dims of above operation
        assert z_comp.get_shape().as_list() == [None, l]

        # Concatenate elements for feed into LSTM
        z_i = tf.concat(1,[hp_i, z_comp])

        # Check dims of LSTM input
        assert z_i.get_shape().as_list() == [None, 2*l]

        # Return resultant hr and state from super class (BasicLSTM) run with z_i as input and current state given to our cell
        hr, state = super(MatchLSTMCell, self).__call__(z_i, state)

        return hr, state

class Encoder(object):
    def __init__(self, size, vocab_dim, FLAGS):
        self.size = size
        self.vocab_dim = vocab_dim
        self.FLAGS = FLAGS

    def init_params(self, l, P, Q, reverse = False):
        if (reverse):
            P = self.FLAGS.max_question_size
            Q = self.FLAGS.max_paragraph_size
        self.WQ = tf.get_variable("WQ", [l,l], initializer=tf.uniform_unit_scaling_initializer(1.0))
        self.WP = tf.get_variable("WP", [l,l], initializer=tf.uniform_unit_scaling_initializer(1.0))
        self.WR = tf.get_variable("WR", [l,l], initializer=tf.uniform_unit_scaling_initializer(1.0))

        self.bP = tf.Variable(tf.zeros([1, l]))
        self.w = tf.Variable(tf.zeros([l,1]))
        self.b = tf.Variable(tf.zeros([1,1]))

        params = [self.WQ, self.WP, self.WR, self.bP, self.w, self.b]
        return params

    def encode(self, input_question, input_paragraph, question_length, paragraph_length, dropout_rate, encoder_state_input = None):    # LSTM Preprocessing and Match-LSTM Layers
        """
        Description:
        """

        assert input_question.get_shape().as_list() == [None, self.FLAGS.max_question_size, self.FLAGS.embedding_size]
        assert input_paragraph.get_shape().as_list() == [None, self.FLAGS.max_paragraph_size, self.FLAGS.embedding_size]

        input_question = tf.nn.dropout(input_question, dropout_rate)
        input_paragraph = tf.nn.dropout(input_paragraph, dropout_rate)

        #Preprocessing LSTM
        with tf.variable_scope("question_encode"):
            cell = tf.nn.rnn_cell.BasicLSTMCell(self.size) #self.size passed in through initialization from "state_size" flag
            HQ, _ = tf.nn.dynamic_rnn(cell, input_question, sequence_length = question_length,  dtype = tf.float32)

        with tf.variable_scope("paragraph_encode"):
            cell2 = tf.nn.rnn_cell.BasicLSTMCell(self.size)
            HP, _ = tf.nn.dynamic_rnn(cell2, input_paragraph, sequence_length = paragraph_length, dtype = tf.float32)   #sequence length masks dynamic_rnn

        assert HQ.get_shape().as_list() == [None, self.FLAGS.max_question_size, self.FLAGS.state_size]
        assert HP.get_shape().as_list() == [None, self.FLAGS.max_paragraph_size, self.FLAGS.state_size]

        #HQ = tf.nn.dropout(HQ, dropout_rate)    # A dropout rate of 1 indicates no dropout
        #HP = tf.nn.dropout(HP, dropout_rate)

        # Encoding params
        l = self.size
        Q = self.FLAGS.max_question_size
        P = self.FLAGS.max_paragraph_size

        with tf.variable_scope("match_lstm_1"):
            params = self.init_params(l,Q,P)

            # Initialize forward and backward matching LSTMcells with same matching params
            with tf.variable_scope("forward"):
                cell_f = MatchLSTMCell(l, HQ, self.FLAGS, params)

                if (self.FLAGS.deep):
                    cell_f = [cell_f] + [tf.nn.rnn_cell.BasicLSTMCell(self.size)]*2
                    cell_f = tf.nn.rnn_cell.MultiRNNCell(cell_f)

            with tf.variable_scope("backward"):
                cell_b = MatchLSTMCell(l, HQ, self.FLAGS, params)

                if (self.FLAGS.deep):
                    cell_b = [cell_b] + [tf.nn.rnn_cell.BasicLSTMCell(self.size)]*2
                    cell_b = tf.nn.rnn_cell.MultiRNNCell(cell_b)

            # Calculate encodings for both forward and backward directions
            with tf.variable_scope("hp2"):
                (HP2_right, HP2_left), _ = tf.nn.bidirectional_dynamic_rnn(cell_f, cell_b, HP, sequence_length = paragraph_length, dtype = tf.float32)

        #HP2_right = tf.nn.dropout(HP2_right, dropout_rate)
        #HP2_left = tf.nn.dropout(HP2_left, dropout_rate)

        with tf.variable_scope("match_lstm_2"):
            params = self.init_params(l,Q,P, reverse = True)
            with tf.variable_scope("right_question"):
                cell_rq = MatchLSTMCell(l, HP2_right, self.FLAGS, params, reverse = True)

            with tf.variable_scope("left_question"):
                cell_lq = MatchLSTMCell(l, HP2_left, self.FLAGS, params, reverse = True)

            with tf.variable_scope("hq2"):
                (HQ2_right, HQ2_left), _ = tf.nn.bidirectional_dynamic_rnn(cell_rq, cell_lq, HQ, sequence_length = question_length, dtype = tf.float32)

        #HQ2_right = tf.nn.dropout(HQ2_right, dropout_rate)
        #HQ2_left = tf.nn.dropout(HQ2_left, dropout_rate)
        with tf.variable_scope("match_lstm_3"):
            params = self.init_params(l,Q,P)
            with tf.variable_scope("right_para"):
                cell_rp = MatchLSTMCell(l, HQ2_right, self.FLAGS, params)

            with tf.variable_scope("left_para"):
                cell_lp = MatchLSTMCell(l, HQ2_left, self.FLAGS, params)

            with tf.variable_scope("hr"):
                (HR_right, HR_left), _ = tf.nn.bidirectional_dynamic_rnn(cell_rp, cell_lp, HP2_right, sequence_length = paragraph_length, dtype = tf.float32)

        ### Append the two things calculated above into H^R
        HR = tf.concat(2,[HR_right, HR_left])
        assert HR.get_shape().as_list() == [None, P, 2*l]

        dropout_rate2 = (1 - dropout_rate)/2.0 + dropout_rate
        HR = tf.nn.dropout(HR, dropout_rate2)

        return HR

class Decoder(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

    def decode(self, knowledge_rep, paragraph_mask, cell_init):
        """
        param knowledge_rep: it is a representation of the paragraph and question
        return: tuple that contains the logits for the distributions of start and end token
        """

        # Decode Params
        l = self.FLAGS.state_size
        P = self.FLAGS.max_paragraph_size
        Hr = knowledge_rep
        mask = tf.log(tf.cast(paragraph_mask, tf.float32))

        # Decode variables
        V = tf.get_variable("V", [2*l,l], initializer=tf.contrib.layers.xavier_initializer())
        Wa = tf.get_variable("Wa", [l,l], initializer=tf.contrib.layers.xavier_initializer())
        ba = tf.Variable(tf.zeros([1,l]), name = "ba")
        v = tf.Variable(tf.zeros([l,1]), name = "v")
        c = tf.Variable(tf.zeros([1]), name = "c")

        # Basic LSTM for decoding
        cell = tf.nn.rnn_cell.BasicLSTMCell(l)

        # Preds[0] for predictions start span, and Preds[1] for end of span
        preds = [None, None]

        # Initial hidden layer (and state) from placeholder
        hk = cell_init
        cell_state = (hk, hk)
        assert hk.get_shape().as_list() == [None, l]

        # Just two iterations of decoding for the start point and then the end point
        for i, _ in enumerate(preds):
            if i > 0: #Round 2 should reuse variables from before
                tf.get_variable_scope().reuse_variables()

            # Mult and extend using hack to get shape compatable
            term2 = tf.matmul(hk,Wa) + ba
            term2 = tf.tile(tf.expand_dims(term2,1),[1,P,1])
            #term2 = tf.transpose(tf.stack([term2 for _ in range(P)]), [1,0,2])
            assert term2.get_shape().as_list() == [None, P, l]

            # Reshape and matmul
            Hr_shaped = tf.reshape(Hr, [-1, 2*l])
            term1 = tf.matmul(Hr_shaped, V)
            term1 = tf.reshape(term1, [-1, P, l])
            assert term1.get_shape().as_list() == [None, P, l]

            # Add terms and tanh them
            Fk = tf.tanh(term1 + term2)
            assert Fk.get_shape().as_list() == [None, P, l]

            # Generate beta_term v^T * Fk + c * e(P)
            Fk_shaped = tf.reshape(Fk, [-1, l])
            beta_term = tf.matmul(Fk_shaped, v) + c
            beta_term = tf.reshape(beta_term ,[-1, P, 1])
            assert beta_term.get_shape().as_list() == [None, P, 1]

            #TEST OTHER MASK VERSION
            beta_term_masked = tf.squeeze(beta_term,2) + mask
            assert beta_term_masked.get_shape().as_list() == [None, P]

            # Get Beta (prob dist over the paragraph)
            beta = tf.nn.softmax(beta_term_masked)
            beta_shaped = tf.expand_dims(beta, 2)
            assert beta_shaped.get_shape().as_list() == [None, P, 1]

            # Setup input to LSTM
            Hr_shaped_cell = tf.transpose(Hr, [0, 2, 1])
            cell_input = tf.squeeze(tf.batch_matmul(Hr_shaped_cell, beta_shaped), [2])
            assert cell_input.get_shape().as_list() == [None, 2*l]

            # Ouput and State for next iteration
            hk, cell_state = cell(cell_input, cell_state)

            #Save a 2D rep of Beta as output
            preds[i] = beta_term_masked

        return tuple(preds) # Bs, Be [batchsize, paragraph_length]


class QASystem(object):
    def __init__(self, encoder, decoder, FLAGS, *args):
        """
        Initializes your System
        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        self.encoder = encoder
        self.decoder = decoder
        self.FLAGS = FLAGS

        # ==== set up variables ========
        self.learning_rate = tf.Variable(float(self.FLAGS.learning_rate), trainable = False, name = "learning_rate")
        self.global_step = tf.Variable(int(0), trainable = False, name = "global_step")

        # # ==== set up placeholder tokens ======== 3d (because of batching)
        self.paragraph_placeholder = tf.placeholder(tf.int32, (None, self.FLAGS.max_paragraph_size), name="paragraph_placeholder")
        self.question_placeholder = tf.placeholder(tf.int32, (None, self.FLAGS.max_question_size), name="question_placeholder")
        self.start_answer_placeholder = tf.placeholder(tf.int32, (None), name="start_answer_placeholder")
        self.end_answer_placeholder = tf.placeholder(tf.int32, (None), name="end_answer_placeholder")
        self.paragraph_mask_placeholder = tf.placeholder(tf.bool, (None, self.FLAGS.max_paragraph_size), name="paragraph_mask_placeholder")
        self.paragraph_length = tf.placeholder(tf.int32, (None), name="paragraph_length")
        self.question_length = tf.placeholder(tf.int32, (None), name="question_length")
        self.cell_initial_placeholder = tf.placeholder(tf.float32, (None, self.FLAGS.state_size), name="cell_init")
        self.dropout_placeholder = tf.placeholder(tf.float32, (), name="dropout_placeholder")

        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
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
        # Get encoding representation from encode
        with vs.variable_scope("encode"):
            Hr = self.encoder.encode(self.question_embedding, self.paragraph_embedding, self.question_length, self.paragraph_length, self.dropout_placeholder)

        # Get Boundary predictions using decode
        with vs.variable_scope("decode"):
            self.pred_s, self.pred_e = self.decoder.decode(Hr, self.paragraph_mask_placeholder, self.cell_initial_placeholder)

        # If using bidirectional ans-ptr model
        if (self.FLAGS.bi_ans):
            #Dims for reversal
            dims = [False, True, False]
            dims_2 = [False, True]

            with vs.variable_scope("reversed_decode"):

                # Reverse Hr and the paragraph mask along P axis
                reversed_Hr = tf.reverse(Hr, dims)
                reversed_paragraph_mask = tf.reverse(self.paragraph_mask_placeholder, dims_2)

                # Get back start and end predictions (but since input was reversed, the start is the reversed end and viceversa)
                reversed_pred_e2, reversed_pred_s2 = self.decoder.decode(reversed_Hr, reversed_paragraph_mask, self.cell_initial_placeholder)

                # Get preds by reversing back
                pred_s2 = tf.reverse(reversed_pred_s2, dims_2)
                pred_e2 = tf.reverse(reversed_pred_e2, dims_2)

                # Avg the predictions across each direction
                self.pred_s = tf.add(self.pred_s,pred_s2)/2.0
                self.pred_e = tf.add(self.pred_e,pred_e2)/2.0


    def setup_predictions(self):
        with vs.variable_scope("prediction"):
            self.Beta_s = tf.nn.softmax(self.pred_s)
            self.Beta_e = tf.nn.softmax(self.pred_e)


    def setup_loss(self):
        with vs.variable_scope("loss"):
            l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(self.pred_s, self.start_answer_placeholder)
            l2 = tf.nn.sparse_softmax_cross_entropy_with_logits(self.pred_e, self.end_answer_placeholder)
            self.loss = tf.reduce_mean(l1+l2)

            self.train_loss_tb = tf.summary.scalar("train_loss", self.loss)
            self.val_loss_tb = tf.summary.scalar("val_loss", self.loss)


    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        """
        with vs.variable_scope("embeddings"):
            embed_file = np.load(self.FLAGS.embed_path)
            pretrained_embeddings = embed_file['glove']
            embeddings = tf.Variable(pretrained_embeddings, name = "embeddings", dtype=tf.float32, trainable = False)
            self.paragraph_embedding = tf.nn.embedding_lookup(embeddings,self.paragraph_placeholder)
            self.question_embedding = tf.nn.embedding_lookup(embeddings,self.question_placeholder)


    def decode(self, session, qs, ps, q_masks, p_masks):  #Currently still decodes one at a time
        """
        Returns the probability distribution over different positions in the paragraph
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

    def simple_search(self, b_s, b_e):
        a_s = np.argmax(b_s, axis = 1)
        a_e = np.argmax(b_e, axis = 1)
        for i, _ in enumerate(a_s):
            if a_e[i] < a_s[i]:
                if np.max(b_s[i,:]) > np.max(b_e[i,:]):   #Move a_e to a_s b/c a_s has a higher probability
                    a_e[i] = a_s[i]
                else:                           #Move a_s to a_e b/c a_e has a higher probability
                    a_s[i] = a_e[i]
        return a_s, a_e

    def search(self, b_s_batch, b_e_batch): # TODO: batch this
        window_size = self.FLAGS.max_answer_size # based on franks histogram
        a_s_batch = []
        a_e_batch = []
        for b_s, b_e in zip(b_s_batch, b_e_batch):
            a_s, a_e, max_p = 0, 0, 0
            num_elem = len(b_s)
            for start_ind in range(num_elem):
                for end_ind in range(start_ind, min(window_size + start_ind, num_elem)):
                    if(b_s[start_ind]*b_e[end_ind] > max_p):
                        max_p = b_s[start_ind]*b_e[end_ind]
                        a_s = start_ind
                        a_e = end_ind
            a_s_batch.append(a_s)
            a_e_batch.append(a_e)

        return a_s_batch, a_e_batch

    def answer(self, session, question, paragraph, question_mask, paragraph_mask):

        assert(len(question) == len(paragraph) and len(question) == len(question_mask) and len(question) == len(paragraph_mask))

        b_s, b_e = self.decode(session, question, paragraph, question_mask, paragraph_mask)

        a_s, a_e = [], []
        if (self.FLAGS.search):
            a_s, a_e = self.search(b_s, b_e)
        else:
            a_s, a_e = self.simple_search(b_s, b_e)

        assert(len(a_s) == len(a_e))
        assert(len(a_s) == len(question))
        assert(all(isinstance(item, (int,long)) for item in a_s))
        assert(all(isinstance(item, (int,long)) for item in a_e))

        return a_s, a_e

    def evaluate_answer(self, session, dataset, rev_vocab, sample=100, log=False):
        """
        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        our_answers = []
        their_answers = []
        eval_set = random.sample(dataset, sample)

        batches, num_batches = get_batches(eval_set, self.FLAGS.batch_size)

        #for question, question_mask, paragraph, paragraph_mask, span, true_answer in eval_set:
        for batch in batches:
            val_questions, val_question_masks, val_paragraphs, val_paragraph_masks, _, val_true_answers = zip(*batch)
            a_s, a_e = self.answer(session, val_questions, val_paragraphs, val_question_masks, val_paragraph_masks)
            for s, e, paragraph in zip(a_s, a_e, val_paragraphs):
                token_answer = paragraph[s : e + 1]      #The slice of the context paragraph that is our answer

                sentence = [rev_vocab[token] for token in token_answer]
                our_answer = ' '.join(word for word in sentence)
                our_answers.append(our_answer)

            for true_answer in val_true_answers:
                their_answer = ' '.join(word for word in true_answer)
                their_answers.append(their_answer)

        assert(len(our_answers) == len(their_answers))

        f1 = exact_match = total = 0
        answer_tuples = zip(their_answers, our_answers)
        for ground_truth, prediction in answer_tuples:
            total += 1
            exact_match += exact_match_score(prediction, ground_truth)
            f1 += f1_score(prediction, ground_truth)

        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, exact_match, sample))
            logging.info("Samples:")
            for i in xrange(min(10, sample)):
                ground_truth, our_answer = answer_tuples[i]
                logging.info("Ground Truth: {}, Our Answer: {}".format(ground_truth, our_answer))

        return f1, exact_match

    def validate(self,session, batch):
        """
        Does not perform any training update.
        :return:
        """
        val_qs, val_q_masks, val_ps, val_p_masks, val_spans, val_answers = zip(*batch)    # Unzip batch, each returned element is a tuple of lists

        input_feed = {}

        start_answers = [val_span[0] for val_span in list(val_spans)]
        end_answers = [val_span[1] for val_span in list(val_spans)]

        input_feed[self.question_placeholder] = np.array(list(val_qs))
        input_feed[self.paragraph_placeholder] = np.array(list(val_ps))
        input_feed[self.start_answer_placeholder] = np.array(start_answers)
        input_feed[self.end_answer_placeholder] = np.array(end_answers)
        input_feed[self.paragraph_mask_placeholder] = np.array(list(val_p_masks))
        input_feed[self.paragraph_length] = np.sum(list(val_p_masks), axis = 1)   # Sum and make into a list
        input_feed[self.question_length] = np.sum(list(val_q_masks), axis = 1)    # Sum and make into a list
        input_feed[self.dropout_placeholder] = self.FLAGS.dropout
        input_feed[self.cell_initial_placeholder] = np.zeros((len(val_qs), self.FLAGS.state_size))

        output_feed = []

        #output_feed.append(self.train_op)
        output_feed.append(self.loss)
        output_feed.append(self.global_norm)
        output_feed.append(self.global_step)


        if self.FLAGS.tb is True:
            output_feed.append(self.val_loss_tb)
            loss, norm, step, val_tb = session.run(output_feed, input_feed)
            self.tensorboard_writer.add_summary(val_tb, step)
        else:
            loss, norm, step = session.run(output_feed, input_feed)

        return loss, norm, step



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

    def train(self, session, dataset, train_dir, rev_vocab):
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
        model_name = "match-lstm"
        checkpoint_path = os.path.join(train_dir, model_name, rname)
        early_stopping_path = os.path.join(checkpoint_path, "early_stopping")

        train_data = zip(dataset["train_questions"], dataset["train_questions_mask"], dataset["train_context"], dataset["train_context_mask"], dataset["train_span"], dataset["train_answer"])
        val_data = zip(dataset["val_questions"], dataset["val_questions_mask"], dataset["val_context"], dataset["val_context_mask"], dataset["val_span"], dataset["val_answer"])

        #get rid of too long answers
        train_data = [d for d in train_data if d[4][1] < self.FLAGS.max_paragraph_size]

        num_data = len(train_data)
        best_f1 = 0

        # Normal training loop
        rolling_ave_window = 50
        losses = [10]*rolling_ave_window

        val_loss_window = 10
        validate_on_every = 10
        val_losses = [10]*val_loss_window

        for cur_epoch in range(self.FLAGS.epochs):
            batches, num_batches = get_batches(train_data, self.FLAGS.batch_size)
            val_batches, num_val_batches = get_batches(val_data, self.FLAGS.batch_size*2)#*validate_on_every)
            for i, batch in enumerate(batches):
                #Optimatize using batch
                loss, norm, step = self.optimize(session, batch)
                losses[step % rolling_ave_window] = loss
                mean_loss = np.mean(losses)

                #Use current model on val data
                if (i%validate_on_every == 0):
                    val_loss, val_norm, val_step = self.validate(session, val_batches[i%num_val_batches])
                    val_losses[step % val_loss_window] = val_loss
                    mean_val_loss = np.mean(val_losses)

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


