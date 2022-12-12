from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import datetime

import numpy as np

def feature_to_dummy(df, column, drop=False):
    ''' take a serie from a dataframe,
        convert it to dummy and name it like feature_value
        - df is a dataframe
        - column is the name of the column to be transformed
        - if drop is true, the serie is removed from dataframe'''
    tmp = pd.get_dummies(df[column], prefix=column, prefix_sep='_')
    df = pd.concat([df, tmp], axis=1, )
    if drop:
        del df[column]
    return df

def datetime_to_float(d):
    epoch = datetime.datetime.utcfromtimestamp(0)
    total_seconds =  (d - epoch).total_seconds()
    # total_seconds will be in decimals (millisecond precision)
    return total_seconds

if __name__ == "__main__":
    
    data = pd.read_excel('Data_Train.xlsx')
    data = data.drop("Route", axis=1)
    data = data.drop("Additional_Info", axis=1)
    data = data.drop("Dep_Time", axis=1)
    data = data.drop("Arrival_Time", axis=1)
    data = data.drop("Date_of_Journey", axis=1)
    data = data.drop("Duration", axis=1)
    print(data)
    data = feature_to_dummy(data, "Airline", True)
    data = feature_to_dummy(data, "Source", True)
    data = feature_to_dummy(data, "Destination", True)
    data = feature_to_dummy(data, "Total_Stops", True)
    # data["Duration"] = pd.to_timedelta(data["Duration"])
    print(data)
    # print(data.to_numpy())
    
    raw_dataset = data.to_numpy()

    input = raw_dataset[:, :-1]
    outputs = raw_dataset[:, -1]
    
    input_train, input_test, output_train, output_test = train_test_split(
        input, outputs, test_size=0.4
    )

    # reg = LogisticRegression(solver="newton-cg", max_iter=1000)
    reg = tree.DecisionTreeClassifier()
    reg.fit(input_train, output_train)

    train_score = reg.score(input_train, output_train)
    test_score = reg.score(input_test, output_test)

    confusion_matrix_result = confusion_matrix(output_test, reg.predict(input_test))
    print(train_score)
    print(test_score)
    print(confusion_matrix_result)
    
    tree.plot_tree(reg)
    plt.show()
