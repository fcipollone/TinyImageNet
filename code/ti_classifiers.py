import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.python.ops import variable_scope as vs

# Make it easy to switch classifiers using flags
def get_classifier(name, FLAGS):
    if name == AlexNet(FLAGS).name():
        return AlexNet(FLAGS)
    elif name == GoogleNet(FLAGS).name():
        return GoogleNet(FLAGS)
    elif name == ResNet(FLAGS).name():
        return ResNet(FLAGS)
    elif name == DeepResNet(FLAGS).name():
        return DeepResNet(FLAGS)
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

    def forward_pass(self, X, is_training):
        """
        NOTE: Data is in the format NHWC

        param X: A batch of image data
        param is_training: Whether or not this is a training or testing batch
        return: 
        self.raw_scores: a tensor that contains the unsoftmaxed class scores
        """
        raise Exception("NotImplementedError")

    def loss(self, y):
        """
        Setup default loss after setting up the forward pass.
        This is the standard softmax cross entropy loss function
        Must save the output as self.raw_output during the forward pass
        y: the correct labels

        - Returns:
        loss: a double
        """

        l = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=self.raw_scores)
        loss = tf.reduce_mean(l)
        return loss

    def get_optimizer(self, lr):
        if self.FLAGS.optimizer == "adam":
            return tf.train.AdamOptimizer(lr)   # Recommended lr of 1e-3
        elif self.FLAGS.optimizer == 'nesterov':   
            return tf.train.MomentumOptimizer(lr, momentum = 0.9, use_nesterov = True)   # Recommended lr of 0.1, this is what was used in the resnet paper
        elif self.FLAGS.optimizer == 'rmsprop':
            return tf.train.RMSPropOptimizer(lr)    # Recommended lr of 1e-2
        else:
            raise Exception("InvalidOptimizerError")

    def weight_decay(self):
        return layers.l2_regularizer(self.FLAGS.weight_decay)

    def weight_init(self):
        return layers.variance_scaling_initializer(mode='FAN_OUT')


    def train_op(self, lr, step, loss):
        """
        Setup default optimizer after setting up loss
        This is the standard training operation with Adam optimizer, decayed lr, and gradient clipping
        lr: learning rate
        step: global step

        - Returns:
        train_op: a handle on the training operation
        decayed_lr: the current lr after exponential decay
        global_norm: the current global_norm
        """

        optimizer = self.get_optimizer(lr)

        grads_and_vars = optimizer.compute_gradients(loss, tf.trainable_variables())
        grads = [g for g, v in grads_and_vars]
        variables = [v for g, v in grads_and_vars]

        clipped_grads, global_norm = tf.clip_by_global_norm(grads, self.FLAGS.max_gradient_norm)

        # Batch Norm in tensorflow requires this extra dependency
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op = optimizer.apply_gradients(zip(clipped_grads, variables), global_step = step, name = "apply_clipped_grads")

        return train_op, global_norm

################################################################################################################################

# AlexNet, but with reduced capacity
class AlexNet (ImageClassifier):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

    def name(self):
        return "AlexNet"

    def forward_pass(self, X, is_training):
        # Conv Layers
        print("Input Shape:", X.shape)
        nn = layers.conv2d(X, num_outputs=64, kernel_size=7, stride=1, data_format='NHWC', padding='SAME', weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
        nn = layers.conv2d(nn, num_outputs=64, kernel_size=5, stride=1, data_format='NHWC', padding='SAME', weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
        nn = tf.nn.max_pool(nn, [1,2,2,1], strides=[1,2,2,1], padding='SAME', data_format='NHWC')
        nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = is_training)
        print(nn.shape)

        nn = layers.conv2d(nn, num_outputs=128, kernel_size=5, stride=1, data_format='NHWC', padding='SAME', weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
        nn = tf.nn.max_pool(nn, [1,2,2,1], strides=[1,2,2,1], padding='SAME', data_format='NHWC')
        nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = is_training)
        print(nn.shape)
        
        nn = layers.conv2d(nn, num_outputs=256, kernel_size=3, stride=1, data_format='NHWC', padding='SAME', weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
        nn = layers.conv2d(nn, num_outputs=256, kernel_size=3, stride=1, data_format='NHWC', padding='SAME', weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
        nn = tf.nn.max_pool(nn, [1,2,2,1], strides=[1,2,2,1], padding='SAME', data_format='NHWC')
        print(nn.shape)
        
        # Affine Layers
        nn = layers.flatten(nn)
        nn = layers.fully_connected(inputs = nn, num_outputs = 1024, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
        nn = layers.dropout(nn, keep_prob = 0.5, is_training=is_training)
        nn = layers.fully_connected(inputs = nn, num_outputs = 1024, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
        nn = layers.dropout(nn, keep_prob = 0.5, is_training=is_training)
        self.raw_scores = layers.fully_connected(inputs = nn, num_outputs = self.FLAGS.n_classes, activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())

        assert (self.raw_scores.get_shape().as_list() == [None, self.FLAGS.n_classes])
        return self.raw_scores

    def loss(self, y):
        l = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=self.raw_scores)
        loss = tf.reduce_mean(l)
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss + reg_loss


# A net using google inception modules
class GoogleNet (ImageClassifier):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

    def name(self):
        return "GoogleNet"

    def inception_module(self, X, i, a, b, c, d, e, f):
        """
        a, b, c, d, e, f are integer inputs corresponding to the number of filters for various convolutions
        i is the number of the inception module. Used for namespacing
        """

        with vs.variable_scope("inception" + str(i)):
            conv3 = layers.conv2d(X, num_outputs=a, kernel_size=1, stride=1, data_format='NHWC', padding='SAME', scope = "Conv3")
            conv1 = layers.conv2d(X, num_outputs=b, kernel_size=1, stride=1, data_format='NHWC', padding='SAME', scope = "Conv1")
            conv2 = layers.conv2d(X, num_outputs=d, kernel_size=1, stride=1, data_format='NHWC', padding='SAME', scope = "Conv2")
            mp1 = tf.nn.max_pool(X, [1,3,3,1], strides=[1,1,1,1], padding='SAME', data_format='NHWC', name="max_pool")

            conv4 = layers.conv2d(conv1, num_outputs=c, kernel_size=3, stride=1, data_format='NHWC', padding='SAME', scope = "Conv4")
            conv5 = layers.conv2d(conv2, num_outputs=e, kernel_size=5, stride=1, data_format='NHWC', padding='SAME', scope = "Conv5")
            conv6 = layers.conv2d(mp1, num_outputs=f, kernel_size=1, stride=1, data_format='NHWC', padding='SAME', scope = "Conv6")
            out = tf.concat([conv3, conv4, conv5, conv6], axis=3, name="Concat")
        return out

    def auxiliary_stem(self, X, i, is_training):
        with vs.variable_scope("gradient_helper_stem" + str(i)):
            ap1 = tf.nn.avg_pool(X, [1,5,5,1], strides=[1,3,3,1], padding='VALID', data_format='NHWC', name="avg_pool")
            conv1 = layers.conv2d(ap1, num_outputs=128, kernel_size=1, stride=1, data_format='NHWC', padding='SAME', scope = "conv1")
            fv1 = layers.fully_connected(inputs = conv1, num_outputs = 1024, scope = "fc1")
            drop1 = layers.dropout(fv1, keep_prob = 0.7, is_training=is_training)
            flat1 = layers.flatten(drop1)
            stem_out = layers.fully_connected(inputs = flat1, num_outputs = self.FLAGS.n_classes, activation_fn = None, scope = "fc_out")
        return stem_out

    def forward_pass(self, X, is_training):
        # Stem Network
        conv1 = layers.conv2d(X, num_outputs=64, kernel_size=7, stride=1, data_format='NHWC', padding='SAME', scope = "Conv1")
        conv2 = layers.conv2d(conv1, num_outputs=192, kernel_size=3, stride=1, data_format='NHWC', padding='SAME', scope = "Conv2")
        mp1 = tf.nn.max_pool(conv2, [1,3,3,1], strides=[1,2,2,1], padding='SAME', data_format='NHWC', name="max_pool1")

        # Inception Layers. Params taken from GoogleNet Paper. But they shouldnt matter too much
        incept1 = self.inception_module(mp1, 1, 64, 96, 128, 16, 32, 32)
        incept2 = self.inception_module(incept1, 2, 128, 128, 192, 32, 96, 64)
        mp2 = tf.nn.max_pool(incept2, [1,3,3,1], strides=[1,2,2,1], padding='SAME', data_format='NHWC', name="max_pool")
        incept3 = self.inception_module(mp2, 3, 192, 96, 208, 16, 48, 64)
        incept4 = self.inception_module(incept3, 4, 160, 112, 224, 24, 64, 64)
        incept5 = self.inception_module(incept4, 5, 128, 128, 256, 24, 64, 64)
        incept6 = self.inception_module(incept5, 6, 112, 144, 288, 32, 64, 64)
        incept7 = self.inception_module(incept6, 7, 256, 160, 320, 32, 128, 128)
        mp3 = tf.nn.max_pool(incept7, [1,3,3,1], strides=[1,2,2,1], padding='SAME', data_format='NHWC', name="max_pool")
        incept8 = self.inception_module(mp3, 8, 256, 160, 320, 32, 128, 128)
        incept9 = self.inception_module(incept8, 9, 384, 192, 384, 48, 128, 128)

        # Classifier Output
        _, H1, W1, _ = incept9.shape
        ap1 = tf.nn.avg_pool(incept9, [1,H1,W1,1], strides=[1,1,1,1], padding='VALID', data_format='NHWC', name="avg_pool")  # Filter size is same as input size
        drop1 = layers.dropout(ap1, keep_prob = 0.6, is_training=is_training)
        flat1 = layers.flatten(drop1)
        self.raw_scores = layers.fully_connected(inputs = flat1, num_outputs = self.FLAGS.n_classes, activation_fn = None, scope = "fc_out")

        # Attach auxiliary output stems for helping grads propogate
        self.stem1_scores = self.auxiliary_stem(incept3, 1, is_training)
        self.stem2_scores = self.auxiliary_stem(incept6, 2, is_training)

        assert self.raw_scores.get_shape().as_list() == [None, self.FLAGS.n_classes]
        return self.raw_scores


    def loss(self, y):
        l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=self.raw_scores)
        l2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=self.stem1_scores)
        l3 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=self.stem2_scores)
        loss = tf.reduce_mean(l1) + 0.3*tf.reduce_mean(l2) + 0.3*tf.reduce_mean(l3)
        return loss


# A 34 Layer Resnet
class ResNet (ImageClassifier):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

    def name(self):
        return "ResNet"

    def ResLayer(self, x, filters, stride = 1, is_training = True, scope = "ResLayer"):
        with vs.variable_scope(scope):
            C = x.get_shape().as_list()[3]
            nn = layers.conv2d(x, num_outputs=filters, kernel_size=3, stride=stride, data_format='NHWC', padding='SAME', \
                activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
            nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn1", activation_fn = None)
            nn = tf.nn.relu(nn)

            nn = layers.conv2d(nn, num_outputs=filters, kernel_size=3, stride=1, data_format='NHWC', padding='SAME', \
                activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
            nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn2", activation_fn = None)

            if stride != 1:
                print("Padding identity mapping to correct size")
                x = tf.nn.avg_pool(x, [1,stride,stride,1], strides=[1,stride,stride,1], padding='VALID', data_format='NHWC')
                x = tf.pad(x, [[0,0], [0,0], [0,0], [(filters-C)//2, (filters-C)//2]])    # This is kind of weird

            nn = x + nn     # Identity mapping plus residual connection
            nn = tf.nn.relu(nn)

            print("Image after " + scope + ":", nn.shape)
            return nn

    def forward_pass(self, X, is_training):
        print("Input image: ", X.shape)
        nn = layers.conv2d(X, num_outputs=64, kernel_size=3, stride=1, data_format='NHWC', padding='SAME', \
            activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
        nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn1", activation_fn = None)
        nn = tf.nn.relu(nn)

        # Residual Layers
        nn = self.ResLayer(nn, 64, is_training = is_training, scope = "ResLayer1")
        nn = self.ResLayer(nn, 64, is_training = is_training, scope = "ResLayer2")
        nn = self.ResLayer(nn, 64, is_training = is_training, scope = "ResLayer3")
        nn = self.ResLayer(nn, 128, is_training = is_training, stride = 2, scope = "ResLayer4")
        nn = self.ResLayer(nn, 128, is_training = is_training, scope = "ResLayer5")
        nn = self.ResLayer(nn, 128, is_training = is_training, scope = "ResLayer6")
        nn = self.ResLayer(nn, 128, is_training = is_training, scope = "ResLayer7")
        nn = self.ResLayer(nn, 256, is_training = is_training, stride = 2, scope = "ResLayer8")
        nn = self.ResLayer(nn, 256, is_training = is_training, scope = "ResLayer9")
        nn = self.ResLayer(nn, 256, is_training = is_training, scope = "ResLayer10")
        nn = self.ResLayer(nn, 256, is_training = is_training, scope = "ResLayer11")
        nn = self.ResLayer(nn, 256, is_training = is_training, scope = "ResLayer12")
        nn = self.ResLayer(nn, 256, is_training = is_training, scope = "ResLayer13")
        nn = self.ResLayer(nn, 512, is_training = is_training, stride = 2, scope = "ResLayer14")
        nn = self.ResLayer(nn, 512, is_training = is_training, scope = "ResLayer15")
        nn = self.ResLayer(nn, 512, is_training = is_training, scope = "ResLayer16")

        # Output Stem
        _, H1, W1, _ = nn.shape
        nn = tf.nn.avg_pool(nn, [1,H1,W1,1], strides=[1,1,1,1], padding='VALID', data_format='NHWC', name="avg_pool")  # Filter size is same as input size
        nn = layers.flatten(nn)
        self.raw_scores = layers.fully_connected(inputs = nn, num_outputs = self.FLAGS.n_classes, \
            activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())

        assert (self.raw_scores.get_shape().as_list() == [None, self.FLAGS.n_classes])
        return self.raw_scores

    def loss(self, y):
        l = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=self.raw_scores)
        loss = tf.reduce_mean(l)
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss + reg_loss


# A 50 Layer Resnet
class DeepResNet (ImageClassifier):
    def __init__(self, FLAGS):
        super().__init__(FLAGS)

    def name(self):
        return "DeepResNet"

    def BottleneckResLayer(self, x, filters1, filters2, stride = 1, is_training = True, scope = "ResLayer"):
        with vs.variable_scope(scope):
            C = x.get_shape().as_list()[3]
            nn = layers.conv2d(x, num_outputs=filters1, kernel_size=1, stride=1, data_format='NHWC', padding='SAME', \
                activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
            nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn1", activation_fn = None)
            nn = tf.nn.relu(nn)

            nn = layers.conv2d(x, num_outputs=filters1, kernel_size=3, stride=stride, data_format='NHWC', padding='SAME', \
                activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
            nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn1", activation_fn = None)
            nn = tf.nn.relu(nn)

            nn = layers.conv2d(nn, num_outputs=filters2, kernel_size=1, stride=1, data_format='NHWC', padding='SAME', \
                activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())

            nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn2", activation_fn = None)

            if stride != 1 or filters2 != C:
                print("Projecting identity mapping to correct size")
                x = tf.nn.avg_pool(x, [1,stride,stride,1], strides=[1,stride,stride,1], padding='SAME', data_format='NHWC')
                x = layers.conv2d(x, num_outputs=filters2, kernel_size=1, stride=1, data_format='NHWC', padding='SAME', \
                    activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())

            nn = x + nn     # Identity mapping plus residual connection
            nn = tf.nn.relu(nn)

            print("Image after " + scope + ":", nn.shape)
            return nn

    def forward_pass(self, X, is_training):
        print("Imput image: ", X.shape)
        nn = layers.conv2d(X, num_outputs=64, kernel_size=3, stride=1, data_format='NHWC', padding='SAME', \
            activation_fn = None, weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())
        nn = layers.batch_norm(nn, decay = 0.9, center = True, scale = True, is_training = is_training, scope = "bn1", activation_fn = None)
        nn = tf.nn.relu(nn)

        # Residual Layers
        nn = self.BottleneckResLayer(nn, 64, 256, is_training = is_training, scope = "ResLayer1")
        nn = self.BottleneckResLayer(nn, 64, 256, is_training = is_training, scope = "ResLayer2")
        nn = self.BottleneckResLayer(nn, 64, 256, is_training = is_training, scope = "ResLayer3")

        nn = self.BottleneckResLayer(nn, 128, 512, is_training = is_training, stride = 2, scope = "ResLayer4")
        nn = self.BottleneckResLayer(nn, 128, 512, is_training = is_training, scope = "ResLayer5")
        nn = self.BottleneckResLayer(nn, 128, 512, is_training = is_training, scope = "ResLayer6")
        nn = self.BottleneckResLayer(nn, 128, 512, is_training = is_training, scope = "ResLayer7")

        nn = self.BottleneckResLayer(nn, 256, 1024, is_training = is_training, stride = 2, scope = "ResLayer8")
        nn = self.BottleneckResLayer(nn, 256, 1024, is_training = is_training, scope = "ResLayer9")
        nn = self.BottleneckResLayer(nn, 256, 1024, is_training = is_training, scope = "ResLayer10")
        nn = self.BottleneckResLayer(nn, 256, 1024, is_training = is_training, scope = "ResLayer11")
        nn = self.BottleneckResLayer(nn, 256, 1024, is_training = is_training, scope = "ResLayer12")
        nn = self.BottleneckResLayer(nn, 256, 1024, is_training = is_training, scope = "ResLayer13")
        
        nn = self.BottleneckResLayer(nn, 512, 2048, is_training = is_training, stride = 2, scope = "ResLayer31")
        nn = self.BottleneckResLayer(nn, 512, 2048, is_training = is_training, scope = "ResLayer32")
        nn = self.BottleneckResLayer(nn, 512, 2048, is_training = is_training, scope = "ResLayer33")

        # Output Stem
        _, H1, W1, _ = nn.shape
        nn = tf.nn.avg_pool(nn, [1,H1,W1,1], strides=[1,1,1,1], padding='VALID', data_format='NHWC', name="avg_pool")  # Filter size is same as input size
        nn = layers.flatten(nn)
        self.raw_scores = layers.fully_connected(inputs = nn, num_outputs = self.FLAGS.n_classes, activation_fn = None, \
            weights_initializer = self.weight_init(), weights_regularizer = self.weight_decay())

        assert (self.raw_scores.get_shape().as_list() == [None, self.FLAGS.n_classes])
        return self.raw_scores

    def loss(self, y):
        l = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=self.raw_scores)
        loss = tf.reduce_mean(l)
        reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        return loss + reg_loss

