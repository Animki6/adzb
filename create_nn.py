import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, ARDRegression, ElasticNet, Lasso
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from alcohol import DataPreprocess

def RandomForest_train_and_plot(x_train, x_test, y_train, y_test):
    reg = RandomForestRegressor(max_depth=2, random_state=0,
                                n_estimators=100).fit(x_train[:], y_train[:])
    reg.score(x_train[:], y_train[:])
    predicted_y = reg.predict(x_test[:])
    pred_y_transposed = np.array([[x] for x in predicted_y])
    print(y_test[:] - pred_y_transposed)
    sns.distplot(y_test[:] - pred_y_transposed)
    plt.show()

def linear_reg_train_and_plot(x_train, x_test, y_train, y_test):
    reg = LinearRegression().fit(x_train[:], y_train[:])
    reg.score(x_train[:], y_train[:])
    predicted_y = reg.predict(x_test[:])
    print(y_test[:] - predicted_y)
    sns.distplot(y_test[:] - predicted_y)
    plt.show()

def Lasso_train_and_plot(x_train, x_test, y_train, y_test):
    reg = Lasso(alpha = .1).fit(x_train[:], y_train[:])
    reg.score(x_train[:], y_train[:])
    predicted_y = reg.predict(x_test[:])
    pred_y_transposed = np.array([[x] for x in predicted_y])
    print(y_test[:] - pred_y_transposed)
    sns.distplot(y_test[:] - pred_y_transposed)
    plt.show()

# ElasticNet_train_and_plot
def ARDR_train_and_plot(x_train, x_test, y_train, y_test):
    reg = ARDRegression().fit(x_train[:], y_train[:])
    reg.score(x_train[:], y_train[:])
    predicted_y = reg.predict(x_test[:])
    pred_y_transposed = np.array([[x] for x in predicted_y])
    print(y_test[:] - pred_y_transposed)
    sns.distplot(y_test[:] - pred_y_transposed)
    plt.show()

def ElasticNet_train_and_plot(x_train, x_test, y_train, y_test):
    reg = ElasticNet(alpha = .5, l1_ratio=0.5).fit(x_train[:], y_train[:])
    reg.score(x_train[:], y_train[:])
    predicted_y = reg.predict(x_test[:])
    pred_y_transposed = np.array([[x] for x in predicted_y])
    print(y_test[:] - pred_y_transposed)
    sns.distplot(y_test[:] - pred_y_transposed)
    plt.show()

if __name__ == '__main__':
    preprocesor = DataPreprocess()
    preprocesor.perform()
    dataset = preprocesor.get_new_set()
    logs_path = "/tmp/tesorflow_logs/example/"
    datasetX, datasetY  = DataPreprocess.split_x_y(dataset)
    x_train, x_test, y_train, y_test = train_test_split(datasetX, datasetY, test_size=0.2)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

    # methods to choose
    
    # linear_reg_train_and_plot(x_train, x_test, y_train, y_test)
    # ARDR_train_and_plot(x_train, x_test, y_train, y_test)
    # Lasso_train_and_plot(x_train, x_test, y_train, y_test)
    # ElasticNet_train_and_plot(x_train, x_test, y_train, y_test)
    RandomForest_train_and_plot(x_train, x_test, y_train, y_test)
