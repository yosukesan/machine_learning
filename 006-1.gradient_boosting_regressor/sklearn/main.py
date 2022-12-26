
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

if __name__ == "__main__":

    ice_cream = pd.read_csv('../../sample_data/ice_creame.csv', skiprows=1)

    x = ice_cream['アイス売り上げ']
    y = ice_cream['来客数']

    params = {'objective': 'regression',
                'min_data' : 1}
    lgbr = lgb.train(params, lgb.Dataset(x.values.reshape(-1, 1), y))
    
    plt.scatter(x, y, label='actual') 
    plt.scatter(x, lgbr.predict(x.values.reshape(-1, 1)), label='predictd') 
    plt.legend()
    plt.show()
