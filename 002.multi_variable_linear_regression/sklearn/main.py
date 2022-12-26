
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __name__ == "__main__":
    
    geo_stats = pd.read_csv('../../sample_data/geo_stats.csv', skiprows=1) 

    # test plot
    #sns.pairplot(geo_stats)
    #plt.show()

    # pre process
    x = geo_stats[['平均気温(deg C)', '降水量(mm)']] 
    y = geo_stats['人口密度(person per km2)']
    #print(x, y)

    # normalise
    xx = x.iloc[:, :].apply(lambda x: (x - x.mean())/x.abs().max())
    yy = (y - y.mean()) / y.abs().max()
    #print(x, y)

    lr = LinearRegression()
    lr.fit(x, y)

    print(lr.predict([[15, 1800]]))

    #fig = plt.figure()
    #ax = Axes3D(fig)
    #ax.scatter3D(x['平均気温(deg C)'], x['降水量(mm)'], lr.predict(x))
    #plt.show()
