import tensorflow as tf

from Tensorflow import MNIST

x = tf.placeholder("float", [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = MNIST.mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print(accuracy.eval(feed_dict={x: MNIST.mnist.test.images, y_: MNIST.mnist.test.labels}))
