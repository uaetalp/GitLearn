import tensorflow as tf
import Mnist_Forward
import os
from tensorflow.examples.tutorials.mnist import input_data

Batch_Size = 200
Learning_Rate_Base = 0.1
Learning_Rate_Decay = 0.99
Regularizer = 0.0001
Steps = 50000
Moving_Average_Decay = 0.99
Model_Save_Path = './model/'
Model_Name = 'mnist_model'


def backward(mnist):
    x = tf.placeholder(tf.float32, [None, Mnist_Forward.Input_Node])
    y_ = tf.placeholder(tf.float32, [None, Mnist_Forward.Output_Node])
    y = Mnist_Forward.forward(x, Regularizer)
    global_step = tf.Variable(0, trainable=None)

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem+tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(Learning_Rate_Base, global_step, mnist.train.num_examples/Batch_Size,
                                           Learning_Rate_Decay, staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(Moving_Average_Decay, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        # 断点续训
        ckpt = tf.train.get_checkpoint_state(Model_Save_Path)  # 存储路径
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(Steps):
            xs, ys = mnist.train.next_batch(Batch_Size)
            _, loss_val, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print('after %d steps training, loss on train batch is %f' % (step, loss_val))
                saver.save(sess, os.path.join(Model_Save_Path, Model_Name), global_step=global_step)


def main():
    mnist = input_data.read_data_sets('./data', one_hot=True)
    backward(mnist)


if __name__ == '__main__':
    main()

