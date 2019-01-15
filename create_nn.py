import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from alcohol import DataPreprocess

if __name__ == '__main__':

    random_state = 42
    np.random.seed(random_state)
    tf.set_random_seed(random_state)

    preprocesor = DataPreprocess()
    preprocesor.perform()
    dataset = preprocesor.get_new_set() #portugalski


    '''
    TODO
    - build network
    - run and test
    '''
    logs_path = "/home/zuza/MI_sem2/ADZB/tesorflow_logs/example/"
    # TODO - complete/implement:
    # train_set, validation_set, test_set = some_split_set_method(dataset)

    # trainX, trainY = DataPreprocess.split_x_y(train_set)
    datasetX, datasetY = DataPreprocess.split_x_y(dataset)

    x_train, x_test, y_train, y_test = train_test_split(datasetX, datasetY, test_size=0.2)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)


    # Parameters
    learning_rate = 0.001
    training_epochs = 4000
    hidden_layer_neurons = 38
    display_step = 100
    n_samples = y_train.shape[1]
    keep_prob = 0.8

    n_parameters = x_train.shape[1]

    n_hidden_layers = 1

    W = [] # weights
    b = [] # biases

    W.append(tf.Variable(tf.random_normal([n_parameters, hidden_layer_neurons], mean=0.5, stddev=0.5)))
    b.append(tf.Variable(tf.random_normal([hidden_layer_neurons], mean=0.5, stddev=0.5)))

    for i in range(n_hidden_layers-1):
        W.append(tf.Variable(tf.random_normal([hidden_layer_neurons, hidden_layer_neurons], mean=0.5, stddev=0.5)))
        b.append(tf.Variable(tf.random_normal([hidden_layer_neurons], mean=0.5, stddev=0.5)))

    W.append(tf.Variable(tf.random_normal([hidden_layer_neurons, n_samples], mean=0.5, stddev=0.5)))
    b.append(tf.Variable(tf.random_normal([n_samples], mean=0.5, stddev=0.5)))

    hLayers = []

    #########################        building NN structure:       ########################################

    x = tf.placeholder(tf.float32, [None, n_parameters])  # input where we will feed one row at a time


    for i in range(n_hidden_layers):
        with tf.name_scope('Hidden_layer'+str(i+1)):
            if i == 0:
                l1 = tf.nn.relu(tf.add(tf.matmul(x, W[i]), b[i]))
            else:
                l1 = tf.nn.relu(tf.add(tf.matmul(hLayers[i-1], W[i]), b[i]))
            l1 = tf.nn.dropout(l1, keep_prob)
            hLayers.append(l1[:])


    # y = tf.nn.softmax(tf.add(tf.matmul(x, W), b))       # NN output
    with tf.name_scope('Output'):
        y = tf.nn.softmax(tf.add(b[-1], tf.matmul(hLayers[-1], W[-1])))
        tf.summary.histogram('output', y)
    # y = tf.reciprocal(1 + tf.exp(-tf.matmul(x, W)))
    # to train network we will need a reference output to construct cost function
    y_ref = tf.placeholder(tf.float32, [None, 1])  #placeholder for trainY, (only one value -> predicted grade)

    # Cost function: Mean squared error# Cost fu
    with tf.name_scope('Cost'):
        # cost = tf.reduce_sum(tf.pow(y_ref - y, 2))/(2*n_samples)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_ref))
        tf.summary.scalar('cost', cost)

    # Gradient descent
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
        # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


    ################3       Initialize variables and tensorflow session    ####################333
    init = tf.global_variables_initializer()
    tf.summary.scalar('cost', cost)
    merged_summary = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init)

        summary_writer = tf.summary.FileWriter(logs_path, graph=sess.graph)

        with tf.name_scope('Training'):

            for i in range(training_epochs):
                sess.run(optimizer, feed_dict={x: x_train, y_ref: y_train}) # Take a gradient descent step using our inputs and labels

                if i % display_step == 0:
                    cc = sess.run(cost, feed_dict={x: x_train, y_ref: y_train})
                    print("Training step:", '%04d' % i, "cost=", "{:.9f}".format(cc))
                    summ = sess.run(merged_summary, feed_dict={x: x_train, y_ref: y_train})
                    summary_writer.add_summary(summ, global_step=i)


                    '''
                    to get current weights:
                    current_W = sess.run(W)
                    '''

            print("Optimization Finished!")
            trained_y = sess.run(y, feed_dict={x: x_train, y_ref: y_train})
            print("Trained y=", trained_y)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                prediction_eval = y - y_ref
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(prediction_eval)
        tf.summary.scalar('accuracy', accuracy)



# TODO : How to Validate?

# TODO : Run session with testX, get output Y and compare it with testY
# something like:
# y = sess.run(y, feed_dict={x: testX})


