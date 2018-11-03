import tensorflow as tf
import numpy as np

# tfrecords是一种二进制文件，可以先将图片和标签制作成该格式的文件。用tfrecords进行数据读取
# 提升内存利用率

# 用tf.train.Example的协议存储训练数据，训练数据的特征用键值对的形式表示
# 如 ‘img_raw’:value, 'label':value 值是Byteslist/FloatList/Int64List
# 用SerializeToString()吧数据序列转化成字符串存储

# 生成tfrecords文件
writer = tf.python_io.TFRecordWriter(tfRecordName)  # 新建一个writer
# for 循环遍历每张图和标签：
    example = tf.train.Example(features=tf.train.Features(feature={
        'img_raw': tf.train.Features(bytes_list=tf.train.BytesList(value=[img_raw])), # 这里放入二进制图片
        'label': tf.train.Features(int64_list=tf.train.Int64List(value=labels))     # 放入图片对应的标签
    }))  # 吧每张图片和标签封装到example中
    writer.write(example.SerializeToString())  # 把example进行序列化,生成tfrecords文件
writer.close()

# 解析tfrecords文件
filename_queue = tf.train.string_input_producer([tfRecord_path])
reader = tf.TFRecordReader()  # 新建一个reader
# 读出的样本存入serialized_example中
_, serialized_example = reader.read(filename_queue)
# 解序列化，图片信息和label都存入features中
features = tf.parse_single_example(serialized_example, features=
                                    {'img_raw': tf.FixedLenFeature([], tf.string)
                                    'label': tf.FixedLenFeature([10], tf.int64)})
img = tf.decode_raw(features['img_raw'], tf.uint8)
img.set_shape([784])
img = tf.cast(img, tf.float32)*(1./255)
label = tf.cast(features['labels'], tf.float32)