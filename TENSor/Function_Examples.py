import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
# 从集合中取全部变量，生成一个列表
tf.get_collection('')

# 列表内对应元素相加
tf.add_n([])

# 把x转换成dtype类型
tf.cast(x, dtype)

# 返回最大值所在的索引号,axis为1代表横向，为0代表纵向
tf.arg_max(x, axis)

# 把字符串按照系统路径的形式重新拼接, 下面语句返回home/name
os.path.join('home', 'name')

# 分割字符串，可以读取出所需的数据，如global_step
string.spilt()

# 在其内定义的节点在计算图g中，绘制计算图中的节点
with tf.Graph().as_default() as g:

# 保存模型
saver = tf.train.Saver()
with tf.Session() as sess:
    for i in range(steps):
        if i % epochs == 0:
            saver.save(sess, os.path.join(Model_Save_Path, Model_Name), global_step=global_step)

# 加载模型
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(Model_Save_Path)  # 存储路径
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

# 实例化可还原滑动平均值的saver
ema = tf.train.ExponentialMovingAverage(Moving_Average_Base)
ema_restore = ema.Variables_to_restore()
saver = tf.train.Saver(ema_restore)

# 准确率计算方法 reduce_mean求均值，将矩阵所有元素按方向求和，然后求平均
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))  # 布尔型数组
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 断点续训
        ckpt = tf.train.get_checkpoint_state(Model_Save_Path)  # 存储路径
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

# numpy将数组以二进制格式读取、存储进磁盘，扩展名为.npy
np.save('名字.npy', Some_Array)
Some_variable =  np.load('名.npy', encoding='').item()  # 默认asc2，latin1, bytes .item()遍历（键值对）

# 返回a的维度
tf.shape(a)

# 将bias加到乘加和上
tf.nn.bias_add(wx, bias)

# -1表示根据其它维度自动调整
tf.reshape(tensor, [-1, m])

np.argsort(list)  # 对列表从小到大排序，返回索引

os.getcwd()  # 返回当前工作目录

os.path.join( , , )  # 拼出整个路径，可引导到特定文件

# 当前目录/vgg16.npy  索引到vgg16.npy文件
vgg16_path = os.path.join(os.getcwd(), 'vgg16.npy')










