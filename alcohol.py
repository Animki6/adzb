#!/usr/bin/python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math


merged_cols = ["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob",
                          "reason", "nursery", "internet"]

class DataPreprocess(object):
    def __init__(self, file_path: str = None):
        data_por = pd.read_csv("../student-por.csv", sep=",")
        data_mat = pd.read_csv("./student-mat.csv", sep=',')
        self.__train_set = pd.merge(data_por, data_mat, on=merged_cols)
        self.non_numerical_cols = self.__train_set.select_dtypes(include=['object']).keys()
        self.compare_x_y() # reduces dataset after merge
        self.compare_non_numeric()
        self.non_numerical_cols = self.__train_set.select_dtypes(include=['object']).keys()
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

    def get_new_set(self, subject: str = 'por'):
        por_only_list = [key for key in self.__new_set.keys() if ('_x' in key) and key != 'G3_x']
        math_only_list = [key for key in self.__new_set.keys() if ('_y' in key) and key != 'G3_y']

        if subject == 'por':
            return self.__new_set.drop(labels=math_only_list, axis=1)
        else:
            return self.__new_set.drop(labels=por_only_list, axis=1)

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

    def merge_to_one(self, new_key: str, key1, key2, keep=None):
        if keep is None:
            self.__train_set[new_key] = (self.__train_set[key1]+self.__train_set[key2])/2
        else:
            self.__train_set[new_key] = self.__train_set[key1] if keep == 1 else self.__train_set[key2]

        self.__train_set = self.__train_set.drop(labels=[key1, key2], axis=1)



    def compare_x_y(self):
        # fig, ax = plt.subplots(10, 1)
        index = 0
        diff_list = []
        for key in self.__train_set.keys():
            if key not in self.non_numerical_cols:
                if ('_' in key) and (key[:-2] not in diff_list) and not('G' in key):
                    difference = abs(self.__train_set[key[:-1]+'x'] - self.__train_set[key[:-1]+'y'])

                    if sum(difference)/len(difference) < 0.1:
                        self.merge_to_one(new_key=key[:-2],
                                          key1=key[:-1]+'x',
                                          key2=key[:-1]+'y')

                    # ax[index].bar(range(len(difference)), difference)
                    # ax[index].set_title(key[:-2], loc='right')
                    index += 1
                    diff_list.append(key[:-2])

    def compare_non_numeric(self):
        # fig, ax = plt.subplots(10, 1)
        index = 0
        diff_list = []
        for key in self.__train_set.keys():
            if key in self.non_numerical_cols:
                if ('_' in key) and (key[:-2] not in diff_list) and not('G' in key):
                    difference = self.__train_set[key[:-1]+'x'] != self.__train_set[key[:-1]+'y']

                    if sum(difference.tolist())/len(difference) < 1:
                        self.merge_to_one(new_key=key[:-2],
                                          key1=key[:-1]+'x',
                                          key2=key[:-1]+'y',
                                          keep=1)

                    # ax[index].bar(range(len(difference)), difference)
                    # ax[index].set_title(key[:-2], loc='right')
                    index += 1
                    diff_list.append(key[:-2])

    def generate_histograms_pre(self, cols=4):
        num_of_plots = len(self.__train_set.keys()) - len(self.non_numerical_cols)

        fig, ax = plt.subplots(math.ceil(num_of_plots/cols), cols)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.8)

        index = 0
        for key in self.__train_set.keys():
            if key not in self.non_numerical_cols:
                current_ax = ax[index // cols][index % cols]
                # current_ax.set_title(key)
                sns.distplot(self.__train_set[key], ax=current_ax, kde=True, kde_kws={'bw': 0.55})
                index += 1

                # sns.countplot(x=key, data=self.__train_set, ax=ax[index//3][index%3])

    def generate_bar_plots_pre(self, cols=4):
        num_of_plots = len(self.non_numerical_cols)

        fig, ax = plt.subplots(math.ceil(num_of_plots/cols), cols)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.8)

        index = 0
        for key in self.__train_set.keys():
            if key in self.non_numerical_cols:
                current_ax = ax[index // cols][index % cols]
                sns.countplot(x=key, data=self.__train_set, ax=current_ax)
                index += 1



    def perform(self):
        for key in self.__train_set.keys():
            if key in self.non_numerical_cols:
                self.parse_values_without_dict(key)
            else:
                self.value_map(key)

        self.merge_series_to_dataframe()
        self.dataframe_to_csv('whole_set.csv')

    @staticmethod
    def split_x_y(data_set, which_g='G3_x'):

        x = data_set[:]
        for key in x.keys():
            if 'G' in key:
                x.drop([key], axis=1)
        y = data_set[[which_g]]

        return x.as_matrix(), y.as_matrix()

    def get_correlation_map(self, pre = True):
        if pre:
            sns.heatmap(self.__train_set.corr(), annot=True, fmt=".2f", cbar=True)
        elif self.__new_set:
            sns.heatmap(self.__new_set.corr(), annot=True, fmt=".2f", cbar=True)


if __name__ == "__main__":
    # data_por = pd.read_csv("../student-por.csv", sep=",")
    # print(data_por.shape)
    #
    # data_mat = pd.read_csv("./student-mat.csv", sep=',')
     # = pd.merge(data_mat, data_por, on='school')
    # file1 = '../student-por.csv'
    # file2 = '../student-mat.csv'
    #
    # preprocesor = DataPreprocess(file1)
    # preprocesor.perform()
    # data_por = preprocesor.get_new_set()
    #
    preprocesor = DataPreprocess()
    preprocesor.get_correlation_map()

    #preprocesor.perform()
    # merged = preprocesor.get_new_set()


    # merged = pd.merge(data_por, data_mat,
    #                   on=["school", "sex", "age", "address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob",
    #                       "reason", "nursery", "internet"])


    # print(merged.shape)
    # with open('merger_dumped', 'w') as f:
    #     f.write(merged.to_csv())

    # plt.figure(figsize=(25, 25))
    # sns.heatmap(merged.corr(), annot=True, fmt=".2f", cbar=True)
    # preprocesor.generate_histograms_pre()
    # preprocesor.generate_bar_plots_pre()
    # preprocesor.compare_non_numeric()
    plt.show()


