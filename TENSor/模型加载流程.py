import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def test(mnist):
    # 绘制计算图中的节点
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])
        # 前向传播计算y
        y = forward(x, w)
        # 实例化可还原滑动平均值的saver
        ema = tf.train.ExponentialMovingAverage(Moving_Average_Base)  # Moving_Average_Decay?
        ema_restore = ema.Variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        # 正确率计算方法 reduce_mean求均值，将矩阵所有元素按方向求和，然后求平均
        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))  # 布尔型数组
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                # 加载ckpt模型
                ckpt = tf.train.get_checkpoint_state(Model_Save_Path)  # 存储路径
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # 计算准确率
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
                    # 打印提示
                    print('after ----------------')
                else:
                    print('No checkpoint file found')
                    return

            def main():
                mnist = input_data.read_data_sets('./data/', one_hot=True)
                test(mnist)
            if __name__ == '__main__':
                main()






















