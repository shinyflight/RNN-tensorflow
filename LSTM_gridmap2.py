import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import time
from bn_lstm import BN_LSTMCell

#print('verbosity:',tf.logging.get_verbosity())
#tf.logging.set_verbosity(tf.logging.FATAL)

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

class LSTM_masking(object):
    def __init__(self, train_path, test_path, num_class, log_every, num_hidden, batch_size, epochs, learning_rate):
        self.train_path = train_path
        self.test_path = test_path
        self.num_class = num_class
        self.log_every = log_every
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.data_load()
        self.create_model()


    def load_data(self, path):
        """
        * input
            path: path of data
        * output
            data: (num_examples, len_sequence, state_dim), 25 states are embedded in 2-dimension
            max_len: maximum length of sequence
        """
        file = open(path, 'r')
        pre_data = [line.split(' ') for line in file.readlines()]
        x = []
        y = []
        seq_len = []

        #find max_len
        max_len = 0
        for row in pre_data:
            if max_len < len(row):
                max_len = len(row)

        for row in pre_data:
            x_row_list = []
            y_row_list = []
            for col in row:
                if col != 'NaN' and col != 'NaN\n' and col != '\n':
                    i = int(int(col) / 5)  # number to matrix index (i,j)
                    j = int(col) % 5
                    embedded = [i, j]
                    x_row_list.append(embedded)
                    y_row_list.append(to_categorical(col, self.num_class).tolist()[0])
            seq_len.append(len(x_row_list))
            x_append = x_row_list[0:-1]
            if max_len-len(x_row_list) > 0:
                x_append += [[0]*2 for i in range(max_len-len(x_row_list))]
            x.append(x_append)
            y_append = y_row_list[1:]
            if max_len-len(y_row_list) > 0:
                y_append += [[0]*self.num_class for i in range(max_len-len(x_row_list))]
            y.append(y_append)

        return np.asarray(x), np.asarray(y), seq_len, max_len

    def data_load(self):
        print('loading training data...')
        self.x_train, self.y_train, self.train_seq_len, train_max_len = self.load_data(self.train_path)
        print('training data is loaded completely')
        print('loading test data...')
        self.x_test, self.y_test, self.test_seq_len, test_max_len = self.load_data(self.test_path)
        print('\rtest data is loaded completely')
        self.max_len = max(train_max_len, test_max_len)


    def build_model(self):
        """
        param x: (batch_size, max_len, state_dim)
        param seq_len: a placeholder which is holding lengths of sequences in a batch
        param weights: (batch_size, num_hidden, num_class)
        param biases: (batch_size, num_class)
        return:
        """

        # Define a lstm cell with tensorflow
        self.is_training = tf.placeholder(tf.bool,None,name='is_training')
        lstm_cell = BN_LSTMCell(self.num_hidden, is_training=self.is_training)

        # sequence length를 담은 vector를 static_rnn에 제공하여 길이가 다른 sequence에 대해 dynamic calculation이 가능하도록 함.
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, self.x, dtype=tf.float32, sequence_length=self.seq_len)
        #outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seq_len)
        #outputs = tf.stack(outputs) # split 해서 넣어서 나온 output을 하나의 list 내에 합쳐줌 (n_step, batch_size, state_dim)
        outputs = tf.transpose(outputs, [1, 0, 2]) # [batch_size, n_step, state_dim] 으로 transpose

        # weights
        weight = tf.get_variable('w', [self.num_hidden, self.num_class], initializer=tf.random_normal_initializer(stddev=1.0))
        bias = tf.get_variable('b', [self.num_class], initializer=tf.constant_initializer(0.0))

        outputs = tf.split(axis=0, num_or_size_splits=self.max_len-1, value=outputs, name='output_split')
        logits = [tf.matmul(tf.squeeze(output), weight) + bias for output in outputs]
        logits = tf.transpose(tf.squeeze(logits), [1, 0, 2])
        pred = tf.multiply(logits, self.mask)
        #pred = tf.nn.softmax(z, dim=-1)
        return pred

    def create_model(self):
        # placeholder
        self.x = tf.placeholder(tf.float32, [None, self.max_len-1, 2], name='input') # batch, seq_len, input_dim
        self.y = tf.placeholder(tf.float32, [None, self.max_len-1, self.num_class], name='target')
        self.seq_len = tf.placeholder(tf.int32, [None], name='seqlen')  # batch 내의 각 sequence의 길이
        self.mask = tf.placeholder(tf.float32, [None, self.max_len-1, self.num_class], name='mask')
        print('building LSTM model..')
        self.pred = self.build_model()
        print('building LSTM model complete!')
        with tf.variable_scope('cost'):
            self.train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y), name='train_loss')
            self.test_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y), name='test_loss')
            self.train_loss_summary = tf.summary.scalar('train_loss', self.train_loss)
            self.test_loss_summary = tf.summary.scalar('test_loss', self.test_loss)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.train_loss)

        with tf.variable_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
            self.train_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='train_acc')
            self.test_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='test_acc')
            self.train_acc_summary = tf.summary.scalar('train_acc', self.train_acc)
            self.test_acc_summary = tf.summary.scalar('test_acc', self.test_acc)


    def train_model(self):
        var_init = tf.global_variables_initializer()
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True
        # config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.config.log_device_placement = False
        self.config.allow_soft_placement = True

        with tf.Session(config=self.config) as sess:
            sess.run(var_init)
            writer = tf.summary.FileWriter("C:/tmp/bn_lstm", sess.graph)

            total_batch = int(len(self.x_train) // self.batch_size)
            test_batch_size = int(len(self.x_test) // total_batch)
            print("Training started...")
            start_time = time.time()
            for epoch in range(self.epochs):
                # np.random.shuffle(self.x_train)
                for step in range(total_batch):
                    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    #run_metadata = tf.RunMetadata()
                    start_ind = step * self.batch_size
                    end_ind = (step + 1) * self.batch_size
                    start_ind_t = step * test_batch_size
                    end_ind_t = (step + 1) * test_batch_size
                    # train
                    #start_mask = time.time()
                    train_masks = self.make_mask(self.x_train[start_ind:end_ind])
                    #print(self.x_train[start_ind:end_ind]][0])
                    #print(train_masks[0]) # check masks values
                    #print('%.2fsec' % (time.time()-start_mask)) # check time to make masks
                    train_feed = {self.x: self.x_train[start_ind:end_ind], self.y: self.y_train[start_ind:end_ind], self.seq_len: self.train_seq_len[start_ind:end_ind], self.mask: train_masks, self.is_training:True}
                    train_acc_summary, train_loss_summary, _ = sess.run([self.train_acc_summary, self.train_loss_summary, self.optimizer], feed_dict=train_feed)#, run_metadata=run_metadata)
                    #writer.add_run_metadata(run_metadata, 'step%d' % step)
                    if step % self.log_every == 0:
                        writer.add_summary(train_loss_summary, (epoch * total_batch + step))
                        writer.add_summary(train_acc_summary, (epoch * total_batch + step))
                        # test
                        test_masks = self.make_mask(self.x_test[start_ind_t:end_ind_t])
                        test_feed = {self.x: self.x_test[start_ind_t:end_ind_t], self.y: self.y_test[start_ind_t:end_ind_t], self.seq_len: self.test_seq_len[start_ind_t:end_ind_t], self.mask: test_masks, self.is_training:False}
                        test_acc_summary, test_loss_summary = sess.run([self.test_acc_summary, self.test_loss_summary], feed_dict=test_feed)
                        writer.add_summary(test_acc_summary, (epoch * total_batch + step))
                        writer.add_summary(test_loss_summary, (epoch * total_batch + step))
                        # print(tf.nn.softmax(self.pred).eval(train_feed)) # check output of the model
                        print('\repoch : %d, batch : %d/%d data, eta : %.2fsec, train_acc : %2f, train_loss : %4f, test_acc : %2f, test_loss : %4f'
                              %((epoch+1), (step+1)*self.batch_size, self.x_train.shape[0], (time.time() - start_time),
                                self.train_acc.eval(train_feed), self.train_loss.eval(train_feed), self.test_acc.eval(test_feed), self.test_loss.eval(test_feed)))
                        start_time = time.time()
            print("Optimization Finished!")
            tf.summary.FileWriter.close(writer)

    def inv(self, x, y):
        return 5 * x + y

    def make_mask(self, x):
        x = np.transpose(x, [1, 0, 2])
        mask_seq = []
        with tf.Graph().as_default():
            ind = tf.placeholder(tf.uint8, None, name='onehot_idx')
            pre_mask_data = tf.one_hot(ind, depth=25, name='onehot')
            with tf.Session(config=self.config) as sess2:
                for step in x:  # step : batch * input_dim
                    mask_batch = []
                    for data in step:  # data : input_dim
                        north = self.inv(data[0] - 1, data[1]) if data[0] - 1 >= 0 and data[0] - 1 <= 4 else None
                        east = self.inv(data[0], data[1] + 1) if data[1] + 1 >= 0 and data[1] + 1 <= 4 else None
                        west = self.inv(data[0], data[1] - 1) if data[1] - 1 >= 0 and data[1] - 1 <= 4 else None
                        south = self.inv(data[0] + 1, data[1]) if data[0] + 1 >= 0 and data[0] + 1 <= 4 else None
                        news = [north, east, west, south]
                        idx = []
                        for i in range(4):
                            if news[i] != None:
                                idx.append(news[i])
                        mask_data = sess2.run(pre_mask_data, feed_dict = {ind: idx})
                        mask = np.sum(mask_data, axis=0)
                        mask_batch.append(mask)
                    mask_seq.append(mask_batch)
        return np.transpose(mask_seq, [1, 0, 2])

if __name__ =='__main__':
    model = LSTM_masking('d:/Projects/data/grid_map_data/train_3_test.txt', 'd:/Projects/data/grid_map_data/test_3_test.txt', 25, 10, 30, 20, 10, 0.001) #  train_path, test_path, num_class, log_every, num_hidden, batch_size, epochs, learning_rate
    model.train_model()
