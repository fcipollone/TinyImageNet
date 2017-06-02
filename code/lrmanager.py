import math


class lrManager(object):
    def __init__(self, FLAGS, num_data):
        self.FLAGS = FLAGS

        # For cyclic lr
        self.T = self.FLAGS.epochs*(math.ceil(num_data/self.FLAGS.batch_size))
        if self.FLAGS.cyclic:
            print("Num steps per cycle: " + str(math.ceil(self.T/self.FLAGS.M)))

        # For standard lr
        decay_every = 10
        self.decay_ratio = 1/float(10)
        self.D = decay_every*(math.ceil(num_data/self.FLAGS.batch_size))
        if not self.FLAGS.cyclic:
            print("Num steps per decay: " + str(self.D))

    def get_lr(self, step):
        if self.FLAGS.cyclic:
            return self.get_cyclic_lr(step)
        else:
            return self.get_standard_lr(step)

    def get_cyclic_lr(self, step):
        T = self.T
        M = self.FLAGS.M
        a = self.FLAGS.learning_rate

        lr = a/2*(math.cos(math.pi*((step-1) % math.ceil(T/M))/math.ceil(T/M))+1)
        return lr

    def get_standard_lr(self, step):
        num_decays = step // self.D
        lr = self.FLAGS.learning_rate * self.decay_ratio ** num_decays
        return lr

    def save_snapshot(self, step):
        if self.FLAGS.cyclic:
            T = self.T
            M = self.FLAGS.M

            if (step) % math.ceil(T/M) == 0 and step != 0:
                return True
        
        return False

    def snapshot_num(self, step):
        T = self.T
        M = self.FLAGS.M

        num = math.floor((step+1) / math.ceil(T/M))
        return num