import math


class lrManager(object):
    def __init__(self, FLAGS, num_data):
        self.FLAGS = FLAGS

        # For cyclic lr
        self.T = self.FLAGS.epochs*(math.ceil(num_data/self.FLAGS.batch_size))

        # For standard lr
        decay_every = 10
        self.decay_ratio = 1/float(10)
        self.D = decay_every*(math.ceil(num_data/self.FLAGS.batch_size))

    def get_lr(step):
        if self.FLAGS.cyclic:
            get_cyclic_lr(step)
        else:
            get_standard_lr(step)

    def get_cyclic_lr(step):
        T = self.T
        M = self.FLAGS.M
        a = self.FLAGS.learning_rate

        lr = a/2*(math.cos(math.pi*((step-1) % math.ceil(T/M))/math.ceil(T/M))+1)
        return lr

    def get_standard_lr(step):
        num_decays = math.floor(step / self.D)
        lr = self.FLAGS.learning_rate * self.decay_ratio ** self.D