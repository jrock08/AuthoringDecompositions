import tensorflow as tf

class Losses(object):
    def __init__(self, collection = None, session=None):
        self.collection = collection
        self.loss_weights_dict = {}
        self.loss_dict = {}
        self.loss_weights = []
        self.losses = []
        self.session = session

        self.full_loss_ = None

        self.other_values = []

    def add_loss(self, loss, name,  weight = 1.0):
        self.loss_dict[name] = loss
        if weight is not None:
            if type(weight) is not tf.Tensor:
                weight = tf.Variable(weight, trainable=False, dtype=tf.float32)
            self.loss_weights.append(weight)
            if name is not None:
                self.loss_weights_dict[name] = weight

            self.losses.append(weight * loss)
            tf.scalar_summary(name, loss, collections=self.collection)
            tf.scalar_summary(name + '_weight', weight*loss, collections=self.collection)
        else:
            tf.scalar_summary(name, loss, collections=self.collection)

    def update_weight(self, name, weight):
        self.loss_weights_dict[name].assign(weight).eval(session=self.session)

    def add_summary(self):
        tf.scalar_summary('loss_full_loss', self.full_loss(), collections=self.collection)

    def get_summaries(self):
        return tf.merge_all_summaries(self.collection[0])

    def add_image_summary(self, tag, image, max_images=3):
        tf.image_summary(tag, image, collections=self.collection, max_images=max_images)

    def full_loss(self):
        if self.full_loss_ is None:
            self.full_loss_ = tf.add_n(self.losses)
        return self.full_loss_

