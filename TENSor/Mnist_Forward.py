import tensorflow as tf

Input_Node = 784
Output_Node = 10
Layer1_Node = 500


# 定义神经网络的输入、参数和输出，前向传播过程
def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)

    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regularizer):
    w1 = get_weight([Input_Node, Layer1_Node], regularizer)
    b1 = get_bias(Layer1_Node)
    y1 = tf.nn.relu(tf.matmul(x, w1)+b1)

    w2 = get_weight([Layer1_Node, Output_Node], regularizer)
    b2 = get_bias(Output_Node)
    y = tf.matmul(y1, w2)+b2
    return y
