import tensorflow as tf
import numpy as np
from keras.utils import to_categorical

def load_data(path):
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

    # find max_len
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
                y_row_list.append(to_categorical(col, 25).tolist()[0])
        seq_len.append(len(x_row_list))
        x_append = x_row_list[0:-1]
        if max_len - len(x_row_list) > 0:
            x_append += [[0] * 2 for i in range(max_len - len(x_row_list))]
        x.append(x_append)
        y_append = y_row_list[1:]
        if max_len - len(y_row_list) > 0:
            y_append += [[0] * 25 for i in range(max_len - len(x_row_list))]
        y.append(y_append)

    return np.asarray(x), np.asarray(y), seq_len, max_len



def inv(x):
    return tf.cast(5 * x[0] + x[1], tf.int32)


def make_mask(x):
    mask_seq = []
    st = 0
    for step in x:  # step : batch * input_dim
        st += 1
        zero = tf.constant(0, dtype=tf.float32, name='0')
        four = tf.constant(4, dtype=tf.float32, name='4')
        mask_batch = []
        for data in range(step.shape[0]):  # data : input_dim
            print('\rstep:%d, data:%d/%d' % (st, data + 1, 3), end="")
            north = inv([step[data][0] - 1, step[data][1]]) if (
            (tf.greater_equal((step[data][0] - 1), zero) is not False) and (
            tf.less_equal((step[data][0] - 1), four) is not False)) else None
            west = inv([step[data][0], step[data][1] - 1]) if (
            (tf.greater_equal((step[data][1] - 1), zero) is not False) and (
            tf.less_equal((step[data][1] - 1), four) is not False)) else None
            east = inv([step[data][0], step[data][1] + 1]) if (
            (tf.greater_equal((step[data][1] + 1), zero) is not False) and (
            tf.less_equal((step[data][1] + 1), four) is not False)) else None
            south = inv([step[data][0] + 1, step[data][1]]) if (
            (tf.greater_equal((step[data][0] + 1), zero) is not False) and (
            tf.less_equal((step[data][0] + 1), four) is not False)) else None
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

x = tf.placeholder(tf.float32, [3, 5, 2], name='input')
print(x)
x = tf.transpose(x, [1, 0, 2],name='input_transpose') # (max_len, batch_size, state_dim)로 transpose
print(x)
x = tf.reshape(x, [-1, 2],name='input_reshape') # [batch_size * state_dim]*sequence length로 reshape
print(x)
x = tf.split(value=x, axis=0, num_or_size_splits=5, name='input_split') # time step별로 (batch_size, state_dim)인 tensor로 쪼갬
print(x)
output = make_mask(x)

with tf.Session() as sess:
    #input = np.array([np.array([[1,3],[2,3],[2,4],[3,4],[4,4]]),np.array([[1,3],[2,3],[2,4],[3,4],[4,4]]),np.array([[1,3],[2,3],[2,4],[3,4],[4,4]])])
    input = load_data('d:/Projects/data/grid_map_data/test_test.txt')

    print(sess.run(output,feed_dict={x:input}))