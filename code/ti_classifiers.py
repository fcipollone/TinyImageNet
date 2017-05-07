import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

# Make it easy to switch classifiers using flags
def get_classifier(name, FLAGS):
    if name is DemoClassifier(None).name():
        return DemoClassifier(FLAGS)
    else:
        raise Exception("InvalidClassifierError")
        

class ImageClassifier(object):
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

    def name(self):
        """
        The name of the classifier. Must Implement This. Make it a single word.
        """
        raise Exception("NotImplementedError")

    def image_classify(self, X, is_training):
        """
        NOTE: Data is in the format NHWC

        param X: A batch of image data
        param is_training: Whether or not this is a training or testing batch
        return: tuple that contains the logits for the distributions of start and end token
        """
        raise Exception("NotImplementedError")


# Just my model from CS231n Assignment 2
class DemoClassifier (ImageClassifier):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

    def name(self):
        return "DemoClassifier"

    def image_classify(self, X, is_training):
        # Conv Layers
        conv1 = tf.contrib.layers.conv2d(X, num_outputs=64, kernel_size=3, stride=1, data_format='NHWC', padding='VALID', scope = "Conv1")
        bn1 = tf.contrib.layers.batch_norm(conv1, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn1")
        mp1 = tf.nn.max_pool(bn1, [1,2,2,1], strides=[1,2,2,1], padding='VALID', data_format='NHWC', name="max_pool1")
        
        conv2 = tf.contrib.layers.conv2d(mp1, num_outputs=64, kernel_size=4, stride=1, data_format='NHWC', padding='VALID', scope = "Conv2")
        bn2 = tf.contrib.layers.batch_norm(conv2, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn2")
        mp2 = tf.nn.max_pool(bn2, [1,2,2,1], strides=[1,2,2,1], padding='VALID', data_format='NHWC', name="max_pool2")
        
        conv3 = tf.contrib.layers.conv2d(mp2, num_outputs=32, kernel_size=5, stride=1, data_format='NHWC', padding='VALID', scope = "Conv3")
        bn3 = tf.contrib.layers.batch_norm(conv3, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn3")
        mp3 = tf.nn.max_pool(bn3, [1,2,2,1], strides=[1,2,2,1], padding='VALID', data_format='NHWC', name="max_pool3")
        
        # Affine Layers
        h1_flat = tf.contrib.layers.flatten(mp3)
        fc1 = tf.contrib.layers.fully_connected(inputs = h1_flat, num_outputs = 512, scope = "fc1")
        raw_scores = tf.contrib.layers.fully_connected(inputs = fc1, num_outputs = self.FLAGS.n_classes, activation_fn = None, scope = "fc2")

        assert raw_scores.get_shape().as_list() == [None, self.FLAGS.n_classes]
        return raw_scores