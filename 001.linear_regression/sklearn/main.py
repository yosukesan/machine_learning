
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import pandas as pd

if __name__=="__main__":

    ice_cream = pd.read_csv("../../sample_data/ice_creame.csv", skiprows=1)
    print(ice_cream)

    # linear regiression
    lr = LinearRegression()
    x = ice_cream["アイス売り上げ"]
    y = ice_cream["来客数"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0) 
    print('train size', x_train.shape, y_train.shape)
    print('test size', x_test.shape, y_test.shape)

    # normalise
    x_train = x_train / x_train.abs().max()
    y_train = y_train / y_train.abs().max()

    x = x.values.reshape(-1, 1)
    x_train = x_train.values.reshape(-1, 1)
    x_test = x_test.values.reshape(-1, 1)

    lr.fit(x_train, y_train)

    # check result
    print("result=", y_test, lr.predict(x_test))

    # check accuracy
    print("accuracy = ", mean_squared_error(y_test, lr.predict(x_test)))

    # plot
    plt.scatter(x, y)
    plt.plot(x_test, lr.predict(x_test))
    plt.show()
