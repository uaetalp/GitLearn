import tensorflow as tf

LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_STEP = 1  # 一般为总的样本数/batch—size

global_step = tf.Variable(0, trainable=False)

# 参数：基础学习率，当前训练轮数，隔多少轮更新一次学习率，学习率更新系数，是否梯式下降
Learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, LEARNING_RATE_STEP,
                                           LEARNING_RATE_DECAY, staircase=True)

w = tf.Variable(tf.constant(5.0, dtype=tf.float32))

loss = tf.square(w+1)

train_step = tf.train.GradientDescentOptimizer(Learning_rate).minimize(loss, global_step=global_step)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        learning_rate_val = sess.run(Learning_rate)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        global_step_val = sess.run(global_step)
        print('after %d iterations training, global step is %f, w is %f, learning rate is %f, loss is %f'
              % (i, global_step_val, w_val, learning_rate_val, loss_val))