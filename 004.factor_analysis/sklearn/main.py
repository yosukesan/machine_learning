
from matplotlib import pyplot as plt

import pandas as pd
from pandas import plotting

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis as FA

if __name__ == "__main__":
    
    score = pd.read_csv('../../sample_data/test_score.csv') 
    #plotting.scatter_matrix(score) 
    #plt.show()

    sc = StandardScaler()
    sc.fit(score)
    z = sc.transform(score)

    n_components = 3
    fa = FA(n_components, max_iter=500)
    fitted = fa.fit_transform(z)

    factor_loading_matrix = fa.components_.T

    res = pd.DataFrame(factor_loading_matrix,
                columns=['factor 1', 'factor 2', 'factor 3'],
                index=[score.columns])

    print(res)

    plotting.scatter_matrix(res)
    plt.show()
