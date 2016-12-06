import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def plot_rates(mnist):
    batch_size = 100
    sgd_title = 'sgd'
    sgd_optimizer = tf.train.GradientDescentOptimizer(0.001)
    #sgd_values = run_model(mnist, 100, sgd_optimizer, batch_size)
    sgd_values = run_multilayer_model(mnist, 1000, sgd_optimizer, batch_size)
    sgd_line, = plt.plot(sgd_values, 'g-', label=sgd_title)

    adam_title = 'adam'
    adam_optimizer = tf.train.AdamOptimizer(0.001)
    #sgd_values = run_model(mnist, 100, sgd_optimizer, batch_size)
    adam_values = run_multilayer_model(mnist, 1000, adam_optimizer, batch_size)
    adam_line, = plt.plot(adam_values, 'r-', label=adam_title)
    """
    gd_title = 'gd'
    gd_optimizer = tf.train.GradientDescentOptimizer(0.8)
    num_samples = mnist.train.labels.shape[0]
    gd_values = run_model(mnist, 400, gd_optimizer, num_samples)
    gd_line, = plt.plot(gd_values, 'r-', label=gd_title)
    """

    """
    momentum_title = 'momentum'
    momentum_optimizer = tf.train.MomentumOptimizer(.5, .3)
    momentum_values = run_model(mnist, 200, momentum_optimizer, batch_size)
    momentum_line, = plt.plot(momentum_values, 'b-', label=momentum_title)
    """
    

    plt.legend(handles=[sgd_line, adam_line])#, momentum_line, gd_line])

    plt.show()

def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def help(mnist, iters, optimizer, batch_size):
    """Train a model, recording the loss at each step.

    Args:
        minst: The dataset.
        iters: The number of iterations to record.
        optimizer: An initialized instantiation of the optimizer to use.
        batch_size: The batch size to use.

    Returns:
        A list of loss values at each step of training.
    """
    n_hidden_1 = 256 # 1st layer number of features
    n_hidden_2 = 256 # 2nd layer number of features
    n_input = 784 # MNIST data input (img shape: 28*28)
    n_classes = 10 # MNIST total classes (0-9 digits)
    x = tf.placeholder(tf.float32, [None, 784])

    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    #layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    #layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    pred = out_layer

    #W = tf.Variable(tf.zeros([784, 10]))

    #b = tf.Variable(tf.zeros([10]))
    #y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 10])


    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y_))
    optimizer = tf.train.AdamOptimizer(learning_rate=.001).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(pred,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    values = []

    for i in range(iters):
        print i
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, c, a, yi, o = sess.run([optimizer, cost, accuracy, y, out_layer], feed_dict={x: batch_xs, y_: batch_ys})

        print "COST"
        print c
        print "ACCURACY"
        print a
        print "Y"
        print y
        print "OUT"
        print o

        #correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #loss = 1-sess.run(
        #    accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}) 
        #print "LOSS"
        #print loss
        #values.append(loss)

    return values


def run_multilayer_model(mnist, iters, optimizer, batch_size):
    """Train a model, recording the loss at each step.

    Args:
        minst: The dataset.
        iters: The number of iterations to record.
        optimizer: An initialized instantiation of the optimizer to use.
        batch_size: The batch size to use.

    Returns:
        A list of loss values at each step of training.
    """
    x = tf.placeholder(tf.float32, [None, 784])

    layer_sizes = [784, 256, 256, 10]
    weights = [tf.Variable(tf.random_normal([layer_sizes[i-1], layer_sizes[i]])) for i in range(1, len(layer_sizes))]
    biases = [tf.Variable(tf.random_normal([layer_sizes[i]])) for i in range(1, len(layer_sizes))]
        
    layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights[0]), biases[0]))
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights[1]), biases[1]))
    out_layer = tf.add(tf.matmul(layer2, weights[2]), biases[2])
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out_layer, y_))
    y = tf.nn.softmax(out_layer)

    train_step = optimizer.minimize(cross_entropy)
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    values = []

    for i in range(iters):
        print i
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost = sess.run([train_step, cross_entropy], feed_dict={x: batch_xs, y_: batch_ys})
        #print "COST"
        #print cost
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        loss = 1-sess.run(
            accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}) 
        print "LOSS"
        print loss
        values.append(loss)

    return values

def run_model(mnist, iters, optimizer, batch_size):
    """Train a model, recording the loss at each step.

    Args:
        minst: The dataset.
        iters: The number of iterations to record.
        optimizer: An initialized instantiation of the optimizer to use.
        batch_size: The batch size to use.

    Returns:
        A list of loss values at each step of training.
    """
    x = tf.placeholder(tf.float32, [None, 784])

    W = tf.Variable(tf.zeros([784, 10]))

    b = tf.Variable(tf.zeros([10]))
    out_layer = tf.matmul(x, W) + b
    y = tf.nn.softmax(out_layer)
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

    train_step = optimizer.minimize(cross_entropy)
    
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    values = []

    for i in range(iters):
        print i
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        print "out_layer"
        print sess.run(out_layer, feed_dict={x: batch_xs, y_: batch_ys})
        print "y"
        print sess.run(y, feed_dict={x: batch_xs, y_: batch_ys})
        print "BATCH YS"
        print batch_ys
        print "y_"
        print sess.run(y_,feed_dict={x: batch_xs, y_: batch_ys})
        print sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys})
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        loss = 1-sess.run(
            accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}) 
        print "LOSS"
        print loss
        values.append(loss)

    return values


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    plot_rates(mnist)
    exit(1)
    run_multilayer_model(mnist, 5000, tf.train.AdamOptimizer(learning_rate=.001), 100)
    help(mnist, 5000, None, 100)

if __name__ == '__main__':
    main()
