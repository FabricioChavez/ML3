import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn import random_projection
import math

def f_pairwise_distances(data):
  
  return np.mean(pairwise_distances(data))
# funcion para obtener la data_pca y data_rp
def reduce_dim(data, method):
    if method == 'PCA':
        pca = PCA()
        pca.fit(data)
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance_ratio)
        target_variance = 0.95
        x_index = np.argmax(cumulative_explained_variance >= target_variance)
        print(f'PCA: {x_index} components explain {target_variance} of the variance')
        # SCREE PLOT
        plt.figure(figsize=(8, 5))
        plt.plot(np.cumsum(explained_variance_ratio), marker='o', linestyle='--')
        plt.axhline(y=0.98, color='g', linestyle='--', label='95% Variance Threshold')
        plt.axvline(x=x_index, color='r', linestyle='--', label='95% Variance Threshold')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Scree Plot')
        plt.grid()
        plt.show()
        plt.close()
        # PCA
        pca = PCA(n_components=x_index)
        data_pca = pca.fit_transform(data)
        return data_pca
    elif method == 'RP':
        epsilon = 1
        n = data.shape[0]
        k = int(np.ceil(20*np.log2(n) / (epsilon**2)))
        print(f'RP: {k} components selected')
        trans_gaussian = random_projection.GaussianRandomProjection(n_components=k)
        data_rp = trans_gaussian.fit_transform(data)
        return data_rp
    else:
        print('Method not found')
        return None