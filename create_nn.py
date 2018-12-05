import tensorflow as tf
import pandas as pd

from alcohol import DataPreprocess


preprocesor = DataPreprocess()
preprocesor.perform()
dataset = preprocesor.get_new_set()


'''
TODO
- divide data set to: tran, validation and test set (method!)
- build network
- run and test
'''


x = tf.placeholder("float", [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10]) # 0-9 digits recognition => 10 classes