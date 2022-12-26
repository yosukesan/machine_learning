
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import pandas as pd

from matplotlib import pyplot as plt

if __name__ == "__main__":

    geo_stats = pd.read_csv('../../sample_data/ice_creame.csv', skiprows=1) 
    regr = RandomForestRegressor(max_depth=2, random_state=0)
    x = geo_stats['アイス売り上げ'].values.reshape(-1, 1)
    y = geo_stats['来客数']
    regr.fit(x, y)

    plt.scatter(x, y, label='actual')
    plt.scatter(x, regr.predict(x), label='predicted')
    plt.legend()
    plt.show()
