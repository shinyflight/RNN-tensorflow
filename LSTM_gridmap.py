import numpy as np
import tensorflow as tf
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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
        self.x_test, self.y_train, self.train_seq_len, test_max_len = self.load_data(self.test_path)
        print('test data is loaded completely')
        self.max_len = max(train_max_len, test_max_len)

    def inv(self, x):
        return tf.cast(5 * x[0] + x[1], tf.int32)

    def make_mask(self, x):
        mask_seq = []
        st=0
        for step in x:  # step : batch * input_dim
            st+=1
            zero = tf.constant(0, dtype=tf.float32, name='0')
            four = tf.constant(4, dtype=tf.float32, name='4')
            mask_batch = []
            for data in range(step.shape[0]):  # data : input_dim
                print('\rstep:%d, data:%d/%d'%(st,data+1,self.batch_size),end="")
                north = self.inv([step[data][0]-1,step[data][1]]) if ((tf.greater_equal((step[data][0]-1),zero) is not False) and (tf.less_equal((step[data][0]-1),four) is not False)) else None
                west = self.inv([step[data][0],step[data][1]-1]) if ((tf.greater_equal((step[data][1]-1),zero) is not False) and (tf.less_equal((step[data][1]-1),four) is not False)) else None
                east = self.inv([step[data][0],step[data][1]+1]) if ((tf.greater_equal((step[data][1]+1),zero) is not False) and (tf.less_equal((step[data][1]+1),four) is not False)) else None
                south = self.inv([step[data][0]+1,step[data][1]]) if ((tf.greater_equal((step[data][0]+1),zero) is not False) and (tf.less_equal((step[data][0]+1),four) is not False)) else None
                news = [north, east, west, south]
                ind_list = []
                for i in range(4):
                    if news[i] != None:
                        ind_list.append([news[i]])
                updates = tf.constant([0] * len(ind_list))
                shape = tf.constant([25, ])
                with tf.name_scope("mask_matrix") as scope:
                    mask_data = tf.scatter_nd(ind_list, updates, shape)
                mask_batch.append(mask_data)
            mask_seq.append(mask_batch)
        output = tf.transpose(mask_seq, [1, 0, 2])
        return tf.cast(output, tf.float32)

    def build_model(self, x, seq_len):
        """
        param x: (batch_size, max_len, state_dim)
        param seq_len: a placeholder which is holding lengths of sequences in a batch
        param weights: (batch_size, num_hidden, num_class)
        param biases: (batch_size, num_class)
        return:
        """
        x = tf.transpose(x, [1, 0, 2],name='input_transpose') # (max_len, batch_size, state_dim)로 transpose
        x = tf.reshape(x, [-1, 2],name='input_reshape') # [batch_size * state_dim]*sequence length로 reshape
        x = tf.split(axis=0, num_or_size_splits=self.max_len-1, value=x, name='input_split') # time step별로 (batch_size, state_dim)인 tensor로 쪼갬

        # Define a lstm cell with tensorflow
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.num_hidden)

        # sequence length를 담은 vector를 static_rnn에 제공하여 길이가 다른 sequence에 대해 dynamic calculation이 가능하도록 함.
        outputs, states = tf.nn.dyna(lstm_cell, x, dtype=tf.float32, sequence_length=seq_len)
        #outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seq_len)
        outputs = tf.stack(outputs) # split 해서 넣어서 나온 output을 하나의 list 내에 합쳐줌 (n_step, batch_size, state_dim)
        outputs = tf.transpose(outputs, [1, 0, 2]) # [batch_size, n_step, state_dim] 으로 transpose

        # weights
        with tf.variable_scope('weights'):
            weight = tf.get_variable('w', [self.num_hidden, self.num_class], initializer=tf.random_normal_initializer(stddev=1.0))
            bias = tf.get_variable('b', [self.num_class], initializer=tf.constant_initializer(0.0))

        outputs = tf.split(axis=0, num_or_size_splits=self.batch_size, value=outputs, name='output_split')
        logits = [tf.matmul(tf.squeeze(output), weight) + bias for output in outputs]
        logits = tf.squeeze(logits)
        # masking
        print('making mask..')
        mask = self.make_mask(x)
        print('\nmask is maded!')
        pred = tf.multiply(logits, mask)
        #pred = tf.nn.softmax(z, dim=-1)
        return pred

    def create_model(self):
        # placeholder
        self.seq_len = tf.placeholder(tf.int32, [self.batch_size], name='batch_seqlen')  # batch 내의 각 sequence의 길이
        self.x = tf.placeholder(tf.float32, [self.batch_size, self.max_len-1, 2], name='input') # batch, seq_len, input_dim
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.max_len-1, self.num_class], name='target')
        print('building LSTM model..')
        pred = self.build_model(self.x, self.seq_len)
        print('building LSTM model complete!')
        with tf.variable_scope('cost'):
            self.train_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y), name='train_loss')
            self.test_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.y), name='test_loss')
            self.train_loss_summary = tf.summary.scalar('train_loss', self.train_loss)
            self.test_loss_summary = tf.summary.scalar('test_loss', self.test_loss)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.train_loss)

        with tf.variable_scope('accuracy'):
            correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
            self.train_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='train_acc')
            self.test_acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='test_acc')
            self.train_acc_summary = tf.summary.scalar('train_acc', self.train_acc)
            self.test_acc_summary = tf.summary.scalar('test_acc', self.test_acc)


    def train_model(self):
        var_init = tf.initialize_all_variables()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.device('/gpu:0'):
            sess = tf.Session(config=config)
        writer = tf.summary.FileWriter("C:/tmp/LSTM", sess.graph)
        sess.run(var_init)

        total_batch = int(len(self.x_train) // self.batch_size)
        print("Training started...")
        for epoch in range(self.epochs):
            # np.random.shuffle(self.x_train)
            for step in range(total_batch):
                start_ind = step * self.batch_size
                end_ind = (step + 1) * self.batch_size
                # train
                train_feed = {self.x: self.x_train[start_ind:end_ind], self.y: self.y_train[start_ind:end_ind], self.seq_len: self.train_seq_len[start_ind:end_ind]}
                train_acc_summary, train_loss_summary, _ = sess.run([self.train_acc_summary, self.train_loss_summary, self.optimizer], feed_dict=train_feed)
                writer.add_summary(train_loss_summary, (epoch * total_batch + step))
                writer.add_summary(train_acc_summary, (epoch * total_batch + step))
                if step % self.log_every == 0:
                    # test
                    test_feed = {self.x: self.x_test[start_ind:end_ind], self.y: self.y_test[start_ind:end_ind], self.seq_len:self.test_seq_len[start_ind:end_ind]}
                    test_acc_summary, test_loss_summary = sess.run([self.test_acc_summary, self.test_loss_summary], feed_dict=test_feed)
                    writer.add_summary(test_acc_summary, (epoch * total_batch + step))
                    writer.add_summary(test_loss_summary, (epoch * total_batch + step))
                    print('\repoch : %d, batch : %d/%d, train_acc : %2f, train_loss : %4f, test_acc : %2f, test_loss : %4f'
                          %((epoch+1), (step+1)*self.batch_size, self.x_train.shape[0],
                            self.train_acc.eval(train_feed), self.train_loss.eval(train_feed), self.test_acc.eval(test_feed), self.test_loss.eval(test_feed)))
        print("Optimization Finished!")
        tf.summary.FileWriter.close(writer)

if __name__ =='__main__':
    model = LSTM_masking('d:/Projects/data/grid_map_data/train.txt', 'd:/Projects/data/grid_map_data/test.txt', 25, 10, 12, 3, 1, 0.001) #  train_path, test_path, num_class, log_every, num_hidden, batch_size, epochs, learning_rate
    model.train_model()