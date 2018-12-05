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
        self.__new_set.to_csv(name,sep=",")

    def print_new_set(self):
        print(self.__new_set)

def main():
    preprocess = DataPreprocess()
    preprocess.parse_two({"GP":1,"MS":0}, "school")
    preprocess.parse_two({"F":1,"M":0}, "sex")
    preprocess.value_map("age")
    preprocess.parse_two({"U":1,"R":0}, "address")
    preprocess.parse_two({"GT3":1,"LE3":0}, "famsize")
    preprocess.parse_two({"T":1,"A":0}, "Pstatus")
    preprocess.value_map("Medu")
    preprocess.value_map("Fedu")
    preprocess.create_new_series_and_map("Mjob")
    preprocess.create_new_series_and_map("Fjob")
    preprocess.create_new_series_and_map("reason")
    preprocess.create_new_series_and_map("guardian")
    preprocess.value_map("traveltime")
    preprocess.parse_two({"yes":1,"no":0}, "schoolsup")
    preprocess.parse_two({"yes":1,"no":0}, "famsup")
    preprocess.parse_two({"yes":1,"no":0}, "paid")
    preprocess.parse_two({"yes":1,"no":0}, "activities")
    preprocess.parse_two({"yes":1,"no":0}, "nursery")
    preprocess.parse_two({"yes":1,"no":0}, "higher")
    preprocess.parse_two({"yes":1,"no":0}, "internet")
    preprocess.parse_two({"yes":1,"no":0}, "romantic")
    preprocess.value_map("famrel")
    preprocess.value_map("freetime")
    preprocess.value_map("goout")
    preprocess.value_map("Dalc")
    preprocess.value_map("Walc")
    preprocess.value_map("health")
    preprocess.value_map("absences")
    preprocess.value_map("G1")
    preprocess.value_map("G2")
    preprocess.value_map("G3")
    preprocess.merge_series_to_dataframe()
    preprocess.print_new_set()

if __name__ == "__main__":
    main()
