
import pandas as pd
from pandas import plotting
import lightgbm as lgb
import matplotlib.pyplot as plt

if __name__ == "__main__":

    geo_stats = pd.read_csv('../../sample_data/geo_stats.csv', skiprows=1)
    x = geo_stats[['年日照時間(h)','平均気温(deg C)','降水量(mm)']]
    y = geo_stats['人口密度(person per km2)']

    params = {'objective': 'regression',
                'min_data' : 1}
    lgbr = lgb.train(params, lgb.Dataset(x.values, y))
 
    print(y, lgbr.predict(x.values))

    plt.plot(y, lgbr.predict(x.values))
    plt.show()
 
    #plotting.scatter_matrix(pd.DataFrame(x.values, lgbr.predict(x.values), columns=x.columns.values),
    #            c=list(range(0, geo_stats.shape[0])))
    #plt.show()

    #plotting.scatter_matrix(pd.DataFrame(x.values, y, columns=x.columns.values),
    #            c=list(range(0, geo_stats.shape[0])))
    #plt.show()
