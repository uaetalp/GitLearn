import numpy as np
import tensorflow as tf

BatchSize = 8
seed = 23455
cost = 1
profit = 9

rng = np.random.RandomState(seed)
X = rng.rand(32, 2)

Y = [[x0+x1+(rng.rand()/10.0-0.05)]for x0, x1 in X]

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
# w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

y = tf.matmul(x, w1)
# y = tf.matmul(a, w2)

# loss_mse = tf.reduce_mean(tf.square(y-y_))
loss = tf.reduce_sum(tf.where(tf.greater(y, y_), cost*(y-y_), profit*(y_-y)))
train_step = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEP = 20000
    for i in range(STEP):
        start = (i*BatchSize) % 32
        end = start+BatchSize
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print('After %d training steps, loss on all data is %g' % (i, total_loss))
    print('w1:\n', sess.run(w1))
    # print('w2:', sess.run(w2))
