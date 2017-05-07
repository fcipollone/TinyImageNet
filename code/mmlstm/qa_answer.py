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

from qa_model import Encoder, QASystem, Decoder
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data
from utils import get_dataset, initialize_model, initialize_vocab, get_normalized_train_dir, pad_inputs, get_batches

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during answering.")
tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 150, "Size of each model layer.")
tf.app.flags.DEFINE_integer("embedding_size", 300, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embedding_size}.npz)")
tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")
tf.app.flags.DEFINE_integer("max_paragraph_size", 300, "The length to cut paragraphs off at. MUST be the same as the model.")   # As per Frank's histogram.
tf.app.flags.DEFINE_integer("max_question_size", 20, "The length to cut question off at. MUST be the same as the model.")   # As per Frank's histogram
tf.app.flags.DEFINE_integer("max_answer_size", 20, "Maximum window of answers to search.")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_bool("search", False, "Whether to use advanced search methods")
tf.app.flags.DEFINE_bool("bi_ans", False, "Whether to use advanced bidirectional ans-ptr method")
tf.app.flags.DEFINE_bool("deep", False, "Whether to use advanced deep layer between match and ans-ptr")

FLAGS = tf.app.flags.FLAGS

def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                qustion_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]

                context_data.append(' '.join(context_ids))
                query_data.append(' '.join(qustion_ids))
                question_uuid_data.append(question_uuid)

    return context_data, query_data, question_uuid_data


def prepare_dev(prefix, dev_filename, vocab):
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)

    return context_data, question_data, question_uuid_data


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

    questions_padded, questions_masked = pad_inputs(val_questions, FLAGS.max_question_size)
    context_padded, context_masked = pad_inputs(val_context, FLAGS.max_paragraph_size)

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

    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)

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
