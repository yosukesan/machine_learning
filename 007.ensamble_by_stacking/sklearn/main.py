#https://qiita.com/hara_tatsu/items/336f9fff08b9743dc1d2

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression #重回帰分析
from sklearn.ensemble import RandomForestRegressor #random forest
import lightgbm as lgb
from sklearn.metrics import mean_squared_error as mse

import matplotlib.pyplot as plt

def read_data(input_file_path):
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(input_file_path, skiprows=1)
    #print(df)
    
    train = df['アイス売り上げ']
    test = df['来客数']

    x_train, x_test, y_train, y_test = train_test_split(train, test, test_size=0.25, random_state=0)
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=0)

    print(x_train.shape)
    print(x_valid.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_valid.shape)
    print(y_test.shape)

    return x_train, x_test, y_train, y_test, x_valid, y_train, y_valid, train, test
    
if __name__ == "__main__":

    input_file_path = '../../sample_data/ice_creame.csv'
    x_train, x_test, y_train, y_test, x_valid, y_train, y_valid, train, test = read_data(input_file_path)

    lr = LinearRegression() 
    rf = RandomForestRegressor()
    lg = lgb.LGBMRegressor()

    x_train = x_train.values.reshape(-1, 1)

    lr.fit(x_train, y_train)
    rf.fit(x_train, y_train)
    lg.fit(x_train, y_train)

    x_test = x_test.values.reshape(-1, 1)

    lr_pred = lr.predict(x_test)
    rf_pred = rf.predict(x_test) 
    lg_pred = lg.predict(x_test)

    print(mse(y_test, lr_pred))
    print(mse(y_test, rf_pred))
    print(mse(y_test, lg_pred))

    x_valid = x_valid.values.reshape(-1, 1)

    # first stage predictors
    lr_ppred = lr.predict(x_valid) 
    rf_ppred = rf.predict(x_valid) 
    lg_ppred = lg.predict(x_valid) 

    # メタモデルの特徴量
    stack_pred = np.column_stack((lr_ppred, rf_ppred, lg_ppred))
    meta_model = LinearRegression()
    meta_model.fit(stack_pred, y_valid)
    
    # 事前に予測しておいた値でスタッキングの精度を確認する
    stack_test_pred = np.column_stack((lr_ppred, rf_ppred, lg_ppred))
    meta_test_pred = meta_model.predict(stack_test_pred)
    print(mse(y_test, meta_test_pred))

    plt.plot(y_test, meta_test_pred, color='b')
    plt.scatter(test, train, color='r')
    plt.show()
