import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
BATCH_SIZE = 30
seed = 2

rdm = np.random.RandomState(seed)
X = rdm.randn(300, 2)
Y_ = [int(x0*x0+x1*x1 < 2)for x0, x1 in X]

Y_c = [['red' if y else 'blue'] for y in Y_]
# 对数据集X和标签Y进行shape整理，第一个元素为-1， 表示随第二个元素计算得到
# 第二个元素表示列数，X为n行2列，Y为n行1列
X = np.vstack(X).reshape(-1, 2)
Y_ = np.vstack(Y_).reshape(-1, 1)


# 定义神经网络的输入、参数和输出，前向传播过程
def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b


x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_weight([2, 11], 0.01)
b1 = get_bias([11])
a1 = tf.nn.relu(tf.matmul(x, w1)+b1)

w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
y = tf.matmul(a1, w2)+b2

loss_mse = tf.reduce_mean(tf.square(y-y_))
total_loss = loss_mse+tf.add_n(tf.get_collection('losses'))

# 减小均方差，不含正则项
train_step = tf.train.AdamOptimizer(0.01).minimize(loss_mse)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    steps = 4000
    for i in range(steps):
        start = i*BATCH_SIZE % 300
        end = start+BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 2000 == 0:
            loss_mse_val = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            print('after %d steps training, total loss is %f' % (i, loss_mse_val))
        xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]
        grid = np.c_[xx.ravel(), yy.ravel()]
        probs = sess.run(y, feed_dict={x: grid})
        probs = probs.reshape(xx.shape)

plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.contour(xx, yy, probs, levels=[.5])
plt.show()