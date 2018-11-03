import tensorflow as tf
import numpy as np

batch_size = 8
seed = 1
learning_rate_base = 0.01
decay_rate = 0.99
decay_steps = 32/batch_size

moving_average_decay = 0.99

global_step = tf.Variable(0, trainable=False)

ema = tf.train.ExponentialMovingAverage(moving_average_decay, global_step)
ema_op = ema.apply(tf.trainable_variables())
learning_rate = tf.train.exponential_decay(learning_rate_base, global_step, decay_steps,
                                           decay_rate, staircase=True)

rng = np.random.RandomState(seed)
X = rng.rand(32, 2)
Y = [[int(x0+x1 > 1)] for x0, x1 in X]

x = tf.placeholder(tf.float32, shape=[None, 2])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

w1 = tf.Variable(tf.random_normal([2, 3], stddev=0.1, seed=seed))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=0.1, seed=seed))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

loss = tf.reduce_mean(tf.square(y_-y))
train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss,
                                                                                     global_step=global_step)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    step = 3000
    for i in range(step):
        # sess.run(train_step)
        start = (i*batch_size) % 32

        end = start+batch_size
        sess.run(train_step, feed_dict={x: X[start: end], y_: Y[start: end]})
        if i % 500 == 0:
            loss_total = sess.run(loss, feed_dict={x: X, y_: Y})
            print('after %d steps training, total loss is %f' % (i, loss_total))
    print('w1 is : \n', sess.run(w1))
    print('w2 is : \n', sess.run(w2))