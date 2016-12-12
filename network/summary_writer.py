import tensorflow as tf

class summary_writer():
    def __init__(self, root):
        self.writers = {}
        self.root = root
        self.global_iter = {}

    def add_writer(self, name, graph = None):
        full_name = self.root + '/' + name + '_1'
        self.writers[name] = (1,tf.train.SummaryWriter(full_name, graph = graph))
        self.global_iter[name] = 0

    def increment_writer(self, name):
        val = self.writers[name]
        current_num = val[0]
        val[1].flush()
        val[1].close()
        self.writers[name] = (current_num+1, tf.train.SummaryWriter(self.root + '/' + name + '_' + str(current_num+1)))

    def add_summary(self, name, summary):
        self.writers[name][1].add_summary(summary,global_step=self.global_iter[name])
        self.global_iter[name]+=1

    def flush_all(self):
        for idx in self.writers:
            self.writers[idx][1].flush()

