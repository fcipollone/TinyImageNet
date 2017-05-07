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

def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

def convert_to_vocab_number(filename):
    return_val = []
    if tf.gfile.Exists(filename):
        return_val = []
        with tf.gfile.GFile(filename, mode="rb") as f:
            return_val.extend(f.readlines())
        return_val = [ [int(word) for word in line.strip('\n').split()] for line in return_val]
        return return_val
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def convert_to_vocab_number_except_dont(filename):
    return_val = []
    if tf.gfile.Exists(filename):
        return_val = []
        with tf.gfile.GFile(filename, mode="rb") as f:
            return_val.extend(f.readlines())
        return_val = [ [word for word in line.strip('\n').split()] for line in return_val]
        return return_val
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def pad_inputs(data, max_length):
    padded_data = []
    mask_data = []
    for data in data:
        length = len(data)
        if length >= max_length:
            padded_data.append(data[:max_length])
            mask_data.append([1]*max_length)
        else:
            pad_length = max_length-length
            padded_data.append(data + [0]*pad_length)
            mask_data.append([1]*length + [0]*pad_length)
    return padded_data, mask_data

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

def beta_summaries(var, name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('beta_summaries'):
    mean = tf.reduce_mean(var)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar(name + '_stddev', stddev)
    tf.summary.scalar(name + '_max', tf.reduce_max(var))
    tf.summary.scalar(name + '_min', tf.reduce_min(var))

def get_batches(dataset, batch_size):
    random.shuffle(dataset)
    num_batches = int(math.ceil(len(dataset)/float(batch_size)))
    batches = []
    for i in range(num_batches):
        start_ind = i*batch_size
        end_ind = min(len(dataset),i*batch_size+batch_size)
        batches.append(dataset[start_ind:end_ind])

    return batches, num_batches
