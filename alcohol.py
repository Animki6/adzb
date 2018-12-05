#!/usr/bin/python3

import pandas as pd

class DataPreprocess(object):
    def __init__(self):
        self.__train_set = pd.read_csv("./student-por.csv",sep=";")
        self.__new_set = None
        self.__series = []

    def parse_two(self, value_map, column_name):
        new_series = pd.Series(
            [value_map[item] for item in self.__train_set[column_name]],
            name=column_name)
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

    def perform(self):
        self.parse_two({"GP":1, "MS":0}, "school")
        self.parse_two({"F":1, "M":0}, "sex")
        self.value_map("age")
        self.parse_two({"U":1, "R":0}, "address")
        self.parse_two({"GT3":1, "LE3":0}, "famsize")
        self.parse_two({"T":1, "A":0}, "Pstatus")
        self.value_map("Medu")
        self.value_map("Fedu")
        self.create_new_series_and_map("Mjob")
        self.create_new_series_and_map("Fjob")
        self.create_new_series_and_map("reason")
        self.create_new_series_and_map("guardian")
        self.value_map("traveltime")
        self.parse_two({"yes":1, "no":0}, "schoolsup")
        self.parse_two({"yes":1, "no":0}, "famsup")
        self.parse_two({"yes":1, "no":0}, "paid")
        self.parse_two({"yes":1, "no":0}, "activities")
        self.parse_two({"yes":1, "no":0}, "nursery")
        self.parse_two({"yes":1, "no":0}, "higher")
        self.parse_two({"yes":1, "no":0}, "internet")
        self.parse_two({"yes":1, "no":0}, "romantic")
        self.value_map("famrel")
        self.value_map("freetime")
        self.value_map("goout")
        self.value_map("Dalc")
        self.value_map("Walc")
        self.value_map("health")
        self.value_map("absences")
        self.value_map("G1")
        self.value_map("G2")
        self.value_map("G3")
        self.merge_series_to_dataframe()
        self.dataframe_to_csv('whole_set.csv')

    @staticmethod
    def split_x_y(data_set, which_g='G1'):

        x = data_set[:]
        x.drop(['G1', 'G2', 'G3'], axis=1)
        y = data_set[[which_g]]

        return x.as_matrix(), y.as_matrix()



#
# if __name__ == "__main__":
#     main()
