import tensorflow as tf

# 只有Variable才能被优化
w = tf.Variable(tf.constant(5.0, dtype=tf.float32))

loss = tf.square(w+1)

train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print('after %d iterations, w is %f, loss is %f.' % (i, w_val, loss_val))