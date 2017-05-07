import os
import random
import tensorflow as tf

from os.path import join as pjoin
import logging
import math
import random

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("No checkpoints found in " + train_dir)
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def get_dataset(data_dir, max_question_size, max_paragraph_size):

    train_questions_path = os.path.join(data_dir, "train.ids.question")
    train_answer_path = os.path.join(data_dir, "train.answer")
    train_span_path = os.path.join(data_dir, "train.span")
    train_context_path = os.path.join(data_dir, "train.ids.context")
    val_questions_path = os.path.join(data_dir, "val.ids.question")
    val_answer_path = os.path.join(data_dir, "val.answer")
    val_span_path = os.path.join(data_dir, "val.span")
    val_context_path = os.path.join(data_dir, "val.ids.context")

    train_questions = convert_to_vocab_number(train_questions_path)
    train_span = convert_to_vocab_number(train_span_path)
    train_context = convert_to_vocab_number(train_context_path)
    val_questions = convert_to_vocab_number(val_questions_path)
    val_span = convert_to_vocab_number(val_span_path)
    val_context = convert_to_vocab_number(val_context_path)

    train_answer = convert_to_vocab_number_except_dont(train_answer_path)
    val_answer = convert_to_vocab_number_except_dont(val_answer_path)

    train_questions_padded, train_questions_mask = pad_inputs(train_questions, max_question_size)
    train_context_padded, train_context_mask = pad_inputs(train_context, max_paragraph_size)
    val_questions_padded, val_questions_mask = pad_inputs(val_questions, max_question_size)
    val_context_padded, val_context_mask = pad_inputs(val_context, max_paragraph_size)

    return {"train_questions": train_questions_padded,
            "train_questions_mask":train_questions_mask,
            "train_context": train_context_padded,
            "train_context_mask":train_context_mask,
            "train_answer": train_answer,
            "train_span": train_span,
            "val_questions": val_questions_padded,
            "val_questions_mask":val_questions_mask,
            "val_context": val_context_padded,
            "val_context_mask":val_context_mask,
            "val_answer": val_answer,
            "val_span": val_span}


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def get_batches(dataset, batch_size):
    random.shuffle(dataset)
    num_batches = int(math.ceil(len(dataset)/float(batch_size)))
    batches = []
    for i in range(num_batches):
        start_ind = i*batch_size
        end_ind = min(len(dataset),i*batch_size+batch_size)
        batches.append(dataset[start_ind:end_ind])

    return batches, num_batches
