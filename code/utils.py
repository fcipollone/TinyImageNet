import os
import random
import tensorflow as tf
import math
from os.path import join as pjoin
import logging

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


def get_batches(dataset, batch_size):
    random.shuffle(dataset)
    num_batches = int(math.ceil(len(dataset)/float(batch_size)))
    batches = []
    for i in range(num_batches):
        start_ind = i*batch_size
        end_ind = min(len(dataset),i*batch_size+batch_size)
        batches.append(dataset[start_ind:end_ind])

    return batches, num_batches
