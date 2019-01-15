#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


merged_cols = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob",
                          "reason", "nursery", "internet"]

class DataPreprocess(object):
    def __init__(self, file_path: str = None):
        data_por = pd.read_csv("../student-por.csv", sep=";")
        data_mat = pd.read_csv("./student-mat.csv", sep=';')
        merged = pd.merge(data_por, data_mat, on=merged_cols)
        # self.__train_set = pd.read_csv(file_path)
        self.__train_set = merged
        self.__new_set = None
        self.__series = []

    def parse_two(self, value_map, column_name, new_name):
        """

        :param value_map:
        :param column_name:
        :return:
        """
        new_series = pd.Series(
            [value_map[item] for item in self.__train_set[column_name]],
            name=new_name)
        self.__series.append(new_series)

    def value_map(self, column_name):
        max_v = self.__train_set[column_name].max()
        tS = pd.Series(data=[item/max_v for item in self.__train_set[column_name]],
                  name=column_name)
        self.__series.append(tS)

    def create_new_series_and_map(self, column_name):
        value_set = []
        for item in self.__train_set[column_name]:
            value_set.append(item) if item not in value_set else None

        if len(value_set) == 2:
            value_one = value_set[0]
            data_for_new_series = []
            for x in self.__train_set[column_name]:
                data_for_new_series.append(1) if x == value_one else data_for_new_series.append(0)
            tS = pd.Series(data=data_for_new_series,
                           name=str(column_name + "_" + value_one))
            self.__series.append(tS)
        else:
            for i in value_set:
                data_for_new_series = []
                for x in self.__train_set[column_name]:
                    data_for_new_series.append(1) if x == i else data_for_new_series.append(0)
                tS = pd.Series(data=data_for_new_series,
                          name=str(column_name+"_"+i))
                self.__series.append(tS)

    def merge_series_to_dataframe(self):
        self.__new_set = pd.DataFrame(self.__series).transpose()

    def dataframe_to_csv(self, name):
        self.__new_set.to_csv(name, sep=",")

    def print_new_set(self):
        print(self.__new_set)

    def get_new_set(self):
        return self.__new_set

    def parse_values_without_dict(self, column_name):
        new_dict = {}
        set_keys = set()
        iter_value = 0
        new_name = ""
        [set_keys.add(item) for item in self.__train_set[column_name]]
        for key in set_keys:
            new_dict[key] = iter_value
            iter_value = iter_value + 1
        for k in new_dict.keys():
            if new_dict[k] == 1:
                new_name = "{col_name}_{key}".format(col_name=column_name, key=k)
        if len(set_keys) == 2:
            self.parse_two(new_dict, column_name, new_name)
        else:
            self.create_new_series_and_map(column_name)

    def perform(self):

        non_numerical_cols = self.__train_set.select_dtypes(include=['object']).keys()

        for key in self.__train_set.keys():
            if key in non_numerical_cols:
                self.parse_values_without_dict(key)
            else:
                self.value_map(key)

        self.merge_series_to_dataframe()
        self.dataframe_to_csv('whole_set.csv')

    @staticmethod
    def split_x_y(data_set, which_g='G1_x'):

        x = data_set[:]
        for key in x.keys():
            if 'G' in key:
                x.drop([key], axis=1)
        y = data_set[[which_g]]

        return x.as_matrix(), y.as_matrix()




if __name__ == "__main__":
    preprocesor = DataPreprocess()
    preprocesor.perform()
    merged = preprocesor.get_new_set()
    print(merged.shape)
    with open('merger_dumped', 'w') as f:
        f.write(merged.to_csv())

    plt.figure(figsize=(25, 25))
    sns.heatmap(merged.corr(), annot=True, fmt=".2f", cbar=True)
    plt.show()
