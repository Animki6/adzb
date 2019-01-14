import tensorflow as tf
import pandas as pd

from alcohol import DataPreprocess


preprocesor = DataPreprocess()
preprocesor.perform()
dataset = preprocesor.get_new_set()


'''
TODO
- divide data set to: train, validation and test set (method!)
- build network
- run and test
'''
# TODO - complete/implement:
# train_set, validation_set, test_set = some_split_set_method(dataset)

trainX, trainY = DataPreprocess.split_x_y(train_set)

# Parameters
learning_rate = 0.000001
training_epochs = 2000
display_step = 50
n_samples = trainY.size

n_parameters = trainX.size[0]


#########################        building NN structure:       ########################################

x = tf.placeholder(tf.float32, [None, n_parameters])  # input where we will feed one row at a time

W = tf.Variable(tf.zeros([n_parameters, 1]))     # Weights matrix

b = tf.Variable(tf.zeros([n_parameters]))        # Biases - nie wiem po co ale są

y = tf.nn.softmax(tf.add(tf.matmul(x, W), b))       # NN output

# to train network we will need a reference output to construct cost function
y_ref = tf.placeholder(tf.float32, [None, 1])  #placeholder for trainY, (only one value -> predicted grade)

# Cost function: Mean squared error# Cost fu
cost = tf.reduce_sum(tf.pow(y_ref - y, 2))/(2*n_samples)
# Gradient descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)


################3       Initialize variables and tensorflow session    ####################333
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)


for i in range(training_epochs):
    sess.run(optimizer, feed_dict={x: trainX, y_ref: trainY}) # Take a gradient descent step using our inputs and labels

    if i % display_step == 0:
        cc = sess.run(cost, feed_dict={x: trainX, y_ref: trainY})
        print("Training step:", '%04d' % i, "cost=", "{:.9f}".format(cc))

        '''
        to get current weights:
        current_W = sess.run(W)
        '''

print("Optimization Finished!")
training_cost = sess.run(cost, feed_dict={x: trainX, y_ref: trainY})
print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')


# TODO : How to Validate?

# TODO : Run session with testX, get output Y and comare it with testY
# something like:
# y = sess.run(y, feed_dict={x: testX})