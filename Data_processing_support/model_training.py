from Cluster_Models.Kmeans import KMeans
from Cluster_Models.Agglomerate_method import AgglomerativeMethod
import pandas as pd
# funcion para entrenar el modelo
def train_model(data_train,dataframe, method, num_cluster = 20,random_state = 42,dir_truelabels = None):
    labels = []
    true_labels = []
    pred_labels = []
    if method == 'KMeans':
        kmeans = KMeans(num_cluster=num_cluster,random_state=random_state)
        labels = kmeans.fit(data_train)
    elif method == 'Agglomerative':
        cluster_method = AgglomerativeMethod(data=data_train  , cluster_number=num_cluster , linkage_method='centroid')
        cluster_method.fit()
        labels = cluster_method.labels

    dataframe['predic_label'] = labels
    pred_labels = dataframe['predic_label']
    if dir_truelabels != None:
        # cargar las etiquetas de los videos
        categorical_labels = pd.read_csv(dir_truelabels)
        categorical_labels.set_index("youtube_id")
        labels_dict = categorical_labels.set_index('youtube_id')['label'].to_dict() 
        categorical_labels = []
        for names in dataframe['video']:
            categorical_labels.append(labels_dict[names])
        dataframe['real_label'] = categorical_labels
        true_labels = dataframe['real_label']
    return true_labels,pred_labels
    