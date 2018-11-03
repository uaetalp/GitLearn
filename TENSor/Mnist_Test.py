import tensorflow as tf
import time
import Mnist_Forward
import Mnist_Backward
from tensorflow.examples.tutorials.mnist import input_data

Test_Interval_Secs = 5


def test(mnist):
    # 绘制计算图中的节点
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, Mnist_Forward.Input_Node])
        y_ = tf.placeholder(tf.float32, [None, Mnist_Forward.Output_Node])
        y = Mnist_Forward.forward(x, None)

        # 实例化可还原滑动平均值的saver
        ema = tf.train.ExponentialMovingAverage(Mnist_Backward.Moving_Average_Decay)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        # 准确率计算方法 reduce_mean求均值，将矩阵所有元素按方向求和，然后求平均
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))  # 布尔型数组
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                # 加载ckpt模型
                ckpt = tf.train.get_checkpoint_state(Mnist_Backward.Model_Save_Path)  # 存储路径
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print('after %s steps, test accuracy is %g' % (global_step, accuracy_score))
                else:
                    print('Model Not Found!')
                    return
                time.sleep(Test_Interval_Secs)


def main():
    mnist = input_data.read_data_sets('./data', one_hot=True)
    test(mnist)


if __name__ == '__main__':
    main()

