from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

from qa_model import ImageClassifier, Model
from preprocessing.maybe_download import maybe_download
import qa_data
from utils import get_dataset

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during answering.")
tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")   # ??
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")

FLAGS = tf.app.flags.FLAGS

def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    return context_data, query_data, question_uuid_data


def generate_answers(sess, model, dataset, rev_vocab):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """

    val_questions = [map(int, dataset["val_questions"][i].split()) for i in xrange(len(dataset["val_questions"]))]
    val_context = [map(int, dataset["val_context"][i].split()) for i in xrange(len(dataset["val_context"]))]

    answers = {}

    unified_dataset = zip(questions_padded, questions_masked, context_padded, context_masked, dataset["val_question_uuids"])
    batches, num_batches = get_batches(unified_dataset, FLAGS.batch_size)

    for batch in tqdm(batches):
        val_questions, val_question_masks, val_paragraphs, val_paragraph_masks, uuids = zip(*batch)
        a_s, a_e = model.answer(sess, val_questions, val_paragraphs, val_question_masks, val_paragraph_masks)
        for s, e, paragraph, uuid in zip(a_s, a_e, val_paragraphs, uuids):
            token_answer = paragraph[s : e + 1]      #The slice of the context paragraph that is our answer

            sentence = [rev_vocab[token] for token in token_answer]
            our_answer = ' '.join(word for word in sentence)
            answers[uuid] = our_answer

    print ("Generated {}/{} answers".format(len(answers), len(dataset["val_question_uuids"])))
    return answers


def main(_):

    FLAGS.embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way

    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)
    context_data, question_data, question_uuid_data = prepare_dev(dev_dirname, dev_filename, vocab)
    dataset = {"val_context": context_data, "val_questions": question_data, "val_question_uuids": question_uuid_data}

    # ========= Model-specific =========
    # You must change the following code to adjust to your model

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size, FLAGS=FLAGS)
    decoder = Decoder(FLAGS=FLAGS)

    qa = QASystem(encoder, decoder, FLAGS)

    with tf.Session() as sess:
        #train_dir = get_normalized_train_dir(FLAGS.train_dir)

        train_dir = FLAGS.train_dir
        print ("train_dir: ", train_dir)
        initialize_model(sess, qa, train_dir)

        print ("Generating Answers")
        answers = generate_answers(sess, qa, dataset, rev_vocab)

        # write to json file to root dir
        print ("Writing to json file")
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False)))


if __name__ == "__main__":
  tf.app.run()
