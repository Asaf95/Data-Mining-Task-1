# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pandas as pd
from datacleaner import autoclean, autoclean_cv
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RepeatedKFold
import models


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = pd.read_csv('Call Sample.csv')
    my_clean_data = autoclean(data)
    print(data)
    X = my_clean_data.copy()
    y = my_clean_data.copy()
    y1 = my_clean_data.copy()
    X = X[['Client', 'Site','Supervisor','Agent','Week']].values
    y = y['Service Time'].values
    y1 = y1['Quality Score']*10
    print(y1)
    y1 = y1.astype(int)
    print(y1)
    predict_ml_models = models.do_all(X,y)
    print(predict_ml_models)
    predict_ml_models1 = models.do_all(X,y1)
    print(predict_ml_models1)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
