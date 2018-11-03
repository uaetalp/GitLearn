import tensorflow as tf

w1 = tf.Variable(0, dtype=tf.float32)

global_step = tf.Variable(0, trainable=False)

Moving_Average_Decay = 0.99
# ema滑动平均的操作
ema = tf.train.ExponentialMovingAverage(Moving_Average_Decay, global_step)
# ema_op代表将ema应用在所有可训练的变量上
ema_op = ema.apply(tf.trainable_variables())

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(tf.assign(w1, 1))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w1, 10))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))