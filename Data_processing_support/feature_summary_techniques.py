import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.decomposition import PCA


def needed_components(matrices, variance_threshold=0.95):
    max_components = 0
    for matrix in matrices:
        pca = PCA()
        pca.fit(matrix)
        total_variance = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(total_variance >= variance_threshold) + 1
        if n_components > max_components:
            max_components = n_components
    return max_components


def apply_pca(matrix, variance_threshold=0.95, number_of_components=None):
    pca = PCA()
    pca.fit(matrix)

    # Obtener el número de componentes necesarios para explicar el 95% de la varianza
    total_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components = np.argmax(total_variance >= variance_threshold) + 1

    # Aplicar PCA con el número óptimo de componentes

    if number_of_components is not None:
        pca = PCA(n_components=number_of_components)
        transformed_matrix = pca.fit_transform(matrix)

        return transformed_matrix, number_of_components
    else:
        pca = PCA(n_components=number_of_components)
        transformed_matrix = pca.fit_transform(matrix)

        return transformed_matrix, n_components


def apply_pca_fixed_components(matrices, variance_threshold=0.95):
    max_components = needed_components(matrices, variance_threshold)
    print("Maximun size of components needed :", max_components)
    transformed_matrices = []
    for matrix in matrices:
        pca = PCA(n_components=max_components)
        transformed_matrix = pca.fit_transform(matrix)
        transformed_matrices.append(transformed_matrix)
    return transformed_matrices


def summary_by_mean(data):
    new_data = []

    for i in range(len(data)):
        meandata = np.mean(data[i], axis=0)
        new_data.append(meandata)

    df = np.array(new_data)

    return df


def get_cluster_number(type_of_data):
    file_name = type_of_data + '.csv'
    categorical_labels = pd.read_csv(file_name)
    unique_labels = np.unique(categorical_labels["label"])
    return len(unique_labels)

def add_true_categories_to_data_frame(df ,   type_of_data  ):
    file_name = type_of_data +'.csv'
    categorical_labels = pd.read_csv(file_name)
    categorical_labels.set_index("youtube_id")
    labels_dict = categorical_labels.set_index('youtube_id')['label'].to_dict()

    categorical_labels = []

    for names in df['video']:
        categorical_labels.append(labels_dict[names])

    df['category'] = categorical_labels

    return df


def get_all_metrics(predicted_labels, true_labels, data):
    silhouette_avg = silhouette_score(data, predicted_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)
    ami = adjusted_mutual_info_score(true_labels, predicted_labels)
    print('----------------------------------------')
    print(f'Silhouette Score: {silhouette_avg}')
    print('----------------------------------------')
    print(f"Adjusted Rand Index: {ari}")
    print('----------------------------------------')
    print(f"Mutual Info Index: {ami}")
