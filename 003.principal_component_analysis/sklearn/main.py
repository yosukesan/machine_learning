#https://qiita.com/maskot1977/items/082557fcda78c4cdb41f

import matplotlib.pyplot as plt

import pandas as pd
from pandas import plotting

import sklearn
from sklearn.decomposition import PCA

if __name__ == "__main__":
    
    geo_stats = pd.read_csv('../../sample_data/geo_stats.csv', skiprows=1) 

    # test plot
    #plotting.scatter_matrix(geo_stats)
    #plt.show()

    data = geo_stats.loc[:, geo_stats.columns!='都道府県']
    normed = data.iloc[:, 1:].apply(lambda x: (x-x.mean())/x.std())

    pca = PCA()
    pca.fit(normed)
    features = pca.transform(normed)
    print(pca.explained_variance_ratio_)

    plotting.scatter_matrix(pd.DataFrame(features), c=list(normed.iloc[:, 0]), alpha=0.5)
    plt.show()
