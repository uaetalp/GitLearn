import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
# 占位符，shape = [1, 224, 224, 3] 一次喂入图片的数量，图片分辨率，通道数
x = tf.placeholder(tf.float32, shape=[Batch_Size, Image_Pixels])

# 将数组以二进制格式读出/写入磁盘,扩展名为.npy
np.save('name.npy', some_array)
some_variable = np.load('name.npy', encodeig='').item()
# encoding可以不写，默认为ASC2

# .item() 遍历（键值对）
data_dict = np.load(vgg16.npy, encoding="latin1").item()
# 读取vgg16中的文件，遍历其内键值对，导出模型参数赋值给data_dict

# 把bias加到乘加和上
tf.nn.bias_add(he, bias)

# 对列表从小到大排序，返回索引值
np.argsort(list)

os.getcwd()  # 返回当前工作目录

os.path.join( , , )  # 拼出整个路径，可引导到特定文件

# 当前目录/vgg16.npy  索引到vgg16.npy文件
vgg16_path = os.path.join(os.getcwd(), 'vgg16.npy')

# tf.spilt(切谁， 怎么切， 在哪个维度切)   0代表横向→，   1代表纵向↓
value = np.zeros([5, 30])
split0, split1, split2 = tf.split(value, [4, 15, 11], 1)
# split0, [5, 4]  split1, [5, 15]

# 平均切
split0, split1, split2 = tf.split(value, num_or_size_splits=3, axis=1)

# 粘贴 tf.concat(值， 在哪个维)
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 0) ==> [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
tf.concat([t1, t2], 1) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]

# red, green, blue = tf.split(输入， 3， 3)
# TF卷积的输入格式： [batch, 长， 宽， 深]

# 实例化图对象
fig = plt.figure('name_of_picture')
# 读入图片
img = io.imread(path_of_picture)
ax = fig.add_subplot(number, number, number) # number分别是包含几行、几列、当前是第几个
ax.bar(number_of_bar, value_of_bar, name_of_bar, width_of_bar, color_of_bar)
ax.set_ylabe('')   # y轴名字

























