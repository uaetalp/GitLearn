import tensorflow as tf

# [[1， 2]]一行两列，[1, 2]两行一列
# x = tf.constant([[0.1, 0.2]])
x = tf.placeholder(tf.float32, shape=(None, 2))
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, mean=0, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, mean=0, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y, feed_dict={x: [[0.7, 0.4], [0.1, 0.2]]}))