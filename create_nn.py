import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_absolute_error

from alcohol import DataPreprocess

if __name__ == '__main__':

    random_state = 42
    np.random.seed(random_state)
    tf.set_random_seed(random_state)

    preprocesor = DataPreprocess()
    preprocesor.perform()
    dataset = preprocesor.get_new_set() #portugalski
    print(dataset.keys())


    logs_path = "/home/zuza/MI_sem2/ADZB/tesorflow_logs/example/"



    # trainX, trainY = DataPreprocess.split_x_y(train_set)
    datasetX, datasetY = DataPreprocess.split_x_y(dataset)

    x_train, x_test, y_train, y_test = train_test_split(datasetX, datasetY, test_size=0.2)
    # x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)


    # ParametersMean absolute error
    learning_rate = 0.001
    training_epochs = 8000
    hidden_layer_neurons = 30
    display_step = 100
    n_y = y_train.shape[1]
    keep_prob_val = 0.8
    n_x = x_train.shape[1]

    W = {
        'h1': tf.Variable(tf.random_normal([n_x, hidden_layer_neurons], mean=0.5, stddev=0.5)),
        'out': tf.Variable(tf.random_normal([hidden_layer_neurons, n_y], mean=0.5, stddev=0.5))
    }

    b = {
        'b1': tf.Variable(tf.random_normal([hidden_layer_neurons], mean=0.5, stddev=0.5)),
        'out': tf.Variable(tf.random_normal([n_y], mean=0.5, stddev=0.5))
    }

    keep_prob = tf.placeholder("float")

    def build_network(x, W, b, keep_prob):
        l1 = tf.add(tf.matmul(x, W['h1']), b['b1'])
        l1 = tf.nn.softmax(l1)
        l1 = tf.nn.dropout(l1, keep_prob)
        output_layer = tf.matmul(l1, W['out']) + b['out']
        output_layer = tf.nn.sigmoid(output_layer)

        return output_layer


    with tf.name_scope('Input'):
        x = tf.placeholder(tf.float32, [None, n_x])  # input where we will feed one row at a time

    with tf.name_scope('Output_reference'):
        y_ref = tf.placeholder(tf.float32, [None, 1])  # placeholder for trainY, (only one value -> predicted grade)



    # y = tf.nn.softmax(tf.add(tf.matmul(x, W), b))       # NN output
    # with tf.name_scope('Output'):
    #     y = tf.nn.softmax(tf.add(b[-1], tf.matmul(hLayers[-1], W[-1])))
    #     tf.summary.histogram('output', y)
    # y = tf.reciprocal(1 + tf.exp(-tf.matmul(x, W)))
    # to train network we will need a reference output to construct cost function
    y = build_network(x, W, b, keep_prob)
    tf.summary.histogram('output_error', tf.subtract(y_ref, y))

    # Cost function: Mean squared error# Cost fu
    with tf.name_scope('Cost'):
        cost = tf.reduce_sum(tf.pow(y_ref - y, 2))/(2*n_y)
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_ref))
        tf.summary.scalar('cost', cost)

    # Gradient descent
    with tf.name_scope('Optimizer'):
        # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


    ################3       Initialize variables and tensorflow session    ####################333
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        merged_summary = tf.summary.merge_all()
        sess.run(init)

        summary_writer = tf.summary.FileWriter('{}/{}_{}_{}'.format(logs_path,
                                                                    learning_rate,
                                                                    training_epochs,
                                                                    hidden_layer_neurons
                                                                    ), graph=sess.graph)

        with tf.name_scope('Training'):

            for i in range(training_epochs):
                _, cc = sess.run([optimizer, cost], feed_dict={x: x_train,
                                                               y_ref: y_train,
                                                               keep_prob: keep_prob_val}) # Take a gradient descent step using our inputs and labels

                if i % display_step == 0:
                    print("Training step:", '%04d' % i, "cost=", "{:.9f}".format(cc))
                    summ = sess.run(merged_summary, feed_dict={x: x_train, y_ref: y_train, keep_prob: keep_prob_val})
                    summary_writer.add_summary(summ, global_step=i)



            print("Optimization Finished!")
            trained_y = sess.run(y, feed_dict={x: x_train, y_ref: y_train, keep_prob: keep_prob_val})
            print("Trained y=", trained_y)


            # validacja
        variance_score = []
        mae = []

        for i in range(10):
            y_predicted = sess.run(y, feed_dict={x: x_test, keep_prob: 1.0})
            y_diff = y_predicted - y_test

            variance_score.append(explained_variance_score(y_test, y_predicted))
            print(variance_score)

            mae.append(mean_absolute_error(y_test, y_predicted))
            print(mae)

        print('Variance: {}, sdtdev: {}'.format(np.mean(variance_score), np.std(variance_score)))
        print('Mae: {}, std: {}'.format(np.mean(mae), np.std(mae)))




        # with tf.name_scope('accuracy'):
        #     with tf.name_scope('correct_prediction'):
        #         prediction_eval = y - y_ref
        #     with tf.name_scope('accuracy'):
        #         accuracy = tf.reduce_mean(prediction_eval)
        # tf.summary.scalar('accuracy', accuracy)



# TODO : How to Validate?

# TODO : Run session with testX, get output Y and compare it with testY
# something like:
# y = sess.run(y, feed_dict={x: testX})


