import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def create_weights(shape):
    '''
    随机初始化权重
    :param shape: 权重张量的形状
    :return: 初始化够的张量
    '''
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.05))


def create_biases(shape):
    '''
    随机初始化偏置项
    :param shape:
    :return:
    '''
    return tf.Variable(tf.constant(0.5, shape=shape))


def create_conv_layer(input_layer,
                      in_channels,
                      filter_size,
                      stride_size,
                      out_channels,
                      use_relu=True):
    '''
    创建卷积层
    :param input_layer: 前一层的输入
    :param in_channels: 输入的channel个数
    :param filter_size: filter的边长
    :param stride_size: stride的步长
    :param out_channels: 输出的channel个数
    :param use_relu: 是否使用relu
    :return: 当前层卷积后的结果
    '''
    weights = create_weights([filter_size, filter_size,
                              in_channels, out_channels])
    biases = create_weights([out_channels])

    layer = tf.nn.conv2d(input=input_layer,
                         filter=weights,
                         strides=[1, stride_size, stride_size, 1],
                         padding='SAME')
    layer += biases

    if use_relu is True:
        layer = tf.nn.relu(layer)

    return layer


def create_max_pooling_layer(input_layer,  #
                             filter_size,  #
                             stride_size):  #
    '''

    :param input_layer: 输出层
    :param filter_size: filter的边长
    :param stride_size: stride的步长
    :return: 卷积层最大池化后的结果
    '''
    input_layer = tf.nn.max_pool(input_layer,
                                 [1, filter_size, filter_size, 1],
                                 [1, stride_size, stride_size, 1],
                                 padding='SAME')
    return input_layer


def create_flatten_layer(layer):
    '''
    把输入的layer进行flatten操作
    :param layer: 输入的layer
    :return: flatten操作后的结果
    '''
    layer_num = layer.get_shape()[1:4].num_elements()
    layer = tf.reshape(layer, [-1, layer_num])

    return layer


def create_fc_layer(layer, in_num, out_num, use_relu):
    '''
    创建全连接层
    :param layer: 输入的层
    :return: 全连接后的输出层
    '''
    weight = create_weights(shape=[in_num, out_num])
    biases = create_biases([out_num])

    layer = tf.matmul(layer, weight) + biases

    if use_relu is True:
        return tf.nn.relu(layer)
    return layer


# 输入的数据，None可以根据batch-size的结果自动调整
x_input = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="x_input")
# 预测结果
y_true = tf.placeholder(dtype=tf.float32, shape=[None, 1])
# 把输入的每张图片的数据进行恢复
X_input = tf.reshape(x_input, shape=[-1, 28, 28, 1])

# 第一个卷积层
layer_conv1 = create_conv_layer(input_layer=X_input,
                                in_channels=1,
                                filter_size=5,
                                stride_size=1,
                                out_channels=32,
                                use_relu=True)
# 第一个卷积层的池化层
layer_pool1 = create_max_pooling_layer(input_layer=layer_conv1,
                                       filter_size=2,
                                       stride_size=2)
# 第二个卷积层
layer_conv2 = create_conv_layer(input_layer=layer_pool1,
                                in_channels=32,
                                filter_size=2,
                                stride_size=1,
                                out_channels=64)
# 第二个卷积层池化后的结果
layer_pool2 = create_conv_layer(input_layer=layer_conv2,
                                in_channels=64,
                                filter_size=2,
                                stride_size=2,
                                out_channels=64,
                                use_relu=True)
# layer_pool2进行flatten展开，得到3136维的列向量
fc_layer1 = create_flatten_layer(layer=layer_pool2)
# 全连接，输出1000维的向量，并经过relu函数激活
fc_layer2 = create_fc_layer(layer=fc_layer1,
                            in_num=3136,
                            out_num=1000,
                            use_relu=True)
# 全连接，输出10维向量，作为最终预测的结果
fc_layer3 = create_fc_layer(layer=fc_layer2,
                            in_num=1000,
                            out_num=10,
                            use_relu=False)

# 预测结果和真实值
y_pred = tf.nn.softmax(fc_layer3)
y_true = tf.placeholder(dtype=tf.int64, shape=[None, 10], name="y_true")

# 计算交叉熵，注意这里的logits是fc_layer3，因为这个函数本身自带有Softmax的操作
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=fc_layer3,
                                                        labels=y_true)
# 损失函数
cost = tf.reduce_mean(cross_entropy)
# 随机梯度下降优化
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
# 多批次计算的时候是二维的，在这里提取出第2维的，作为比较
y_pred_cls = tf.argmax(y_pred, axis=1)
y_true_cls = tf.argmax(y_true, axis=1)
# 计算精准度
correction_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correction_prediction, tf.float32), name="accuracy")

# 存储模型的对象
saver = tf.train.Saver()

sess = tf.Session()

sess.run(tf.global_variables_initializer())
iteration = 1000
batch_size = 50
for i in range(iteration):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    sess.run(optimizer, feed_dict={x_input: batch_xs, y_true: batch_ys})
    print("step:", i + 1)
print("training finish !")

# 把当前计算图存储为my_test_model
saver.save(sess, './my_test_model')
print("model is saved to my_test_model!")
sess.close()
