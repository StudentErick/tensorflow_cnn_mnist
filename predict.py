import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

sess = tf.Session()

saver = tf.train.import_meta_graph('my_test_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
x_input = graph.get_tensor_by_name("x_input:0")
y_true = graph.get_tensor_by_name("y_true:0")
accuracy = graph.get_tensor_by_name("accuracy:0")

xs, ys = mnist.train.next_batch(5000)

print("accuracy:", sess.run(accuracy, feed_dict={x_input: xs, y_true: ys}))

sess.close()
