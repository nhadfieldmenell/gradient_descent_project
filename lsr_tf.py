import tensorflow as tf
import random
import numpy as np


def train_model(optimizer, iters):
    n = 600
    d = 100
    batch_size = 20

    xs = np.random.randn(n, d)
    xs = xs.astype(np.float32)
    x_star = np.random.randn(d, 1)
    x_star = x_star.astype(np.float32)
    x_star = x_star / np.linalg.norm(x_star)
    ys = xs.dot(x_star)
    A = tf.constant(xs)
    w_star = tf.constant(x_star, shape=[d,1])
    w = tf.Variable(tf.zeros([d, 1]))
    x = tf.placeholder(tf.float32, shape=(batch_size, d))
    y_ = tf.placeholder(tf.float32, shape=(batch_size, 1))

    y = tf.matmul(x, w)
    b = tf.matmul(A, w_star)
    loss = tf.reduce_mean(tf.nn.l2_loss(y - y_))
    train_step = optimizer.minimize(loss)

    b_estimate = tf.matmul(A, w)

    total_loss = tf.reduce_mean(tf.nn.l2_loss(b_estimate - b))

    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    for i in range(iters):
        print i
        indices = np.arange(len(xs))
        random.shuffle(indices)
        indices = indices[:batch_size]
        batch_xs = xs[indices]
        batch_ys = ys[indices]
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        print sess.run(total_loss)
    pass

def main():
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    optimizer = tf.train.AdamOptimizer(.01)
    train_model(optimizer, 200)
    exit(1)
    #print sess.run(x_star)
    print sess.run(x_star)
    print sess.run(b)


if __name__ == '__main__':
    main()
