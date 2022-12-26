
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import pandas as pd

if __name__=="__main__":

    ice_cream = pd.read_csv("../../sample_data/ice_creame.csv", skiprows=1)
    print(ice_cream)

    # linear regiression
    lr = LinearRegression()
    x = ice_cream["アイス売り上げ"]
    y = ice_cream["来客数"]

    # normalise
    x = x / x.abs().max()
    x = x.values.reshape(-1, 1)
    y = y / y.abs().max()

    lr.fit(x, y)

    # check result
    print("result=", y, lr.predict(x))

    # check accuracy
    print("accuracy = ", mean_squared_error(y, lr.predict(x)))

    # plot
    plt.scatter(x, y)
    plt.plot(x, lr.predict(x))
    plt.show()
