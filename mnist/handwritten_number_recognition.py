import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers.python.layers import initializers
import numpy as np
max_steps = 1000  # 最大迭代次数
learning_rate = 0.001   # 学习率
dropout = 0.2   # dropout时随机保留神经元的比例
batch_size = 100
data_dir = './MNIST_DATA'   # 样本数据存储的路径
log_dir = './MNIST_LOG'    # 输出日志保存的路径

initializer = initializers.xavier_initializer()
sess = tf.InteractiveSession()

mnist_data = input_data.read_data_sets(data_dir,one_hot=True)
# print(np.shape(mnist_data))



#add placeholder for graph
with tf.variable_scope('input'):
    input_x = tf.placeholder(tf.float32,[None,784],name='input_x')
    input_y = tf.placeholder(tf.float32,[None,10],name='input_y')

#initial weight
def weight_variable(shape,name):
    return tf.get_variable(name=name,shape= shape,dtype= tf.float32,initializer=initializer)

def bias_variable(shape,name):
    return tf.get_variable(name=name,shape=shape,initializer=tf.zeros_initializer)


# 绘制参数变化
def variable_summaries(var):
    with tf.name_scope('summaries'):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        # 计算参数的标准差
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # 用直方图记录参数的分布
        tf.summary.histogram('histogram', var)

def nn_layer(input_tensor,input_dim,output_dim,layer_name,activation=tf.nn.relu):
    with tf.variable_scope(layer_name):
        weights = weight_variable(shape=[input_dim,output_dim], name='weights')
        variable_summaries(weights)
        biases = bias_variable(shape=[output_dim], name='biases')
        variable_summaries(biases)
        project = tf.matmul(input_tensor,weights)+biases
        tf.summary.histogram('linear', project)
        logits = activation(project)
        tf.summary.histogram('logits', logits)
        return logits

hidden1 = nn_layer(input_x, 784, 500, 'layer1')

# 创建dropout层
with tf.name_scope('dropout'):
    dropout_rate = tf.placeholder(tf.float32,name='dropout_rate')
    tf.summary.scalar('dropout_keep_probability', dropout_rate)
    dropped = tf.nn.dropout(hidden1, rate=dropout_rate)


logits = nn_layer(dropped, 500, 10, 'layer2', activation=tf.identity)


# 创建损失函数
with tf.variable_scope('loss'):
    # 计算交叉熵损失（每个样本都会有一个损失）
    diff = tf.nn.softmax_cross_entropy_with_logits(labels=input_y, logits=logits)

    # 计算所有样本交叉熵损失的均值
    cross_entropy = tf.reduce_mean(diff)
    tf.summary.scalar('loss', cross_entropy)


train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


# 计算准确率
with tf.variable_scope('accuracy'):
        # 分别将预测和真实的标签中取出最大值的索引，弱相同则返回1(true),不同则返回0(false)
    correct_prediction = tf.equal(tf.argmax(input_y, 1), tf.argmax(logits, 1))
        # 求均值即为准确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# summaries合并
merged = tf.summary.merge_all()
# 写到指定的磁盘路径中
train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
test_writer = tf.summary.FileWriter(log_dir + '/test')

# 运行初始化所有变量
tf.global_variables_initializer().run()

def feed_dict(train):
    if train:
        xs, ys = mnist_data.train.next_batch(batch_size)
        k = dropout
    else:
        xs, ys = mnist_data.test.images, mnist_data.test.labels
        k = 0
    return {input_x: xs, input_y: ys, dropout_rate: k}


for i in range(max_steps):
    if i % 10 == 0:  # 记录测试集的summary与accuracy
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('Accuracy at step %s: %s' % (i, acc))
    else:  # 记录训练集的summary
        summary, _ = sess.run([merged, train_op], feed_dict=feed_dict(True))
        train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()