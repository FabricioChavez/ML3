import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


# Implementación de clases y métodos de clustering
class HeapCluster:
    def __init__(self, tuple_list):
        self.heap = tuple_list
        self.heap_index = {}
        self.heap_size = 0

    def left(self, i):
        return 2 * i + 1

    def right(self, i):
        return 2 * i + 2

    def parent(self, i):
        return (i - 1) // 2

    def swap(self, index_a, index_b):
        temp = self.heap[index_a]
        self.heap[index_a] = self.heap[index_b]
        self.heap[index_b] = temp

    def swap_dict(self, index_a, index_b):
        key_a = str(self.heap[index_a][1]) + str(self.heap[index_a][2])
        key_b = str(self.heap[index_b][1]) + str(self.heap[index_b][2])
        temp = self.heap_index[key_a]
        self.heap_index[key_a] = self.heap_index[key_b]
        self.heap_index[key_b] = temp

    def min_heapify_down(self, index):
        left = self.left(index)
        right = self.right(index)
        if left < self.heap_size and self.heap[left][0] < self.heap[index][0]:
            min = left
        else:
            min = index
        if right < self.heap_size and self.heap[right][0] < self.heap[min][0]:
            min = right
        if min != index:
            self.swap(index, min)
            self.min_heapify_down(min)

    def min_heapify_up(self, index):
        i = index
        value = self.heap[index][0]
        while i > 0 and value < self.heap[self.parent(i)][0]:
            self.swap(i, self.parent(i))
            i = self.parent(i)

    def init_heap_index(self):
        for i, triplet_of_values in enumerate(self.heap):
            key = str(triplet_of_values[1]) + str(triplet_of_values[2])
            self.heap_index[key] = i

    def build_heap(self):
        self.heap_size = len(self.heap)
        for i in range(self.parent(self.heap_size - 1), -1, -1):
            self.min_heapify_down(i)

    def insert_k_mins(self, distance_tuple, k):
        if self.heap_size < k:
            self.insert(distance_tuple)
        else:
            if distance_tuple[0] < self.heap[0][0]:
                self.heap[0] = distance_tuple
                self.min_heapify_down(0)

    def insert(self, distance_tuple):
        self.heap.append(distance_tuple)
        self.heap_size = len(self.heap)
        self.min_heapify_up(self.heap_size - 1)

    def remove(self, label_a, label_b):
        key = str(label_a) + str(label_b)
        if key not in self.heap_index:
            return
        index_to_remove = self.heap_index[key]
        self.swap_dict(index_to_remove, self.heap_size - 1)
        self.swap(index_to_remove, self.heap_size - 1)
        value = self.heap[index_to_remove][0]
        self.heap.pop()
        self.heap_size -= 1

        if self.heap_size == 0:
            del self.heap_index[key]
            return

        left = self.left(index_to_remove)
        right = self.right(index_to_remove)
        parent = self.parent(index_to_remove)

        if left < self.heap_size and value > self.heap[left][0]:
            self.min_heapify_down(index_to_remove)
        elif right < self.heap_size and value > self.heap[right][0]:
            self.min_heapify_down(index_to_remove)
        elif parent >= 0 and value < self.heap[parent][0]:
            self.min_heapify_up(index_to_remove)

        del self.heap_index[key]

    def pop(self):
        if len(self.heap) == 0:
            return (float('inf'), None, None)

        result = self.heap[0]
        key_to_remove = str(self.heap[0][1]) + str(self.heap[0][2])
        self.swap(0, self.heap_size - 1)
        self.heap.pop()
        self.heap_size -= 1
        self.min_heapify_down(0)
        return result

class Cluster:
    def __init__(self, cluster_label=None, data_point=None, linkage='centroid'):
        self.cluster_label = cluster_label
        self.data_point = data_point
        self.cluster_list = [self.data_point]
        self.linkage = linkage

    def __str__(self):
        return "Current cluster is " + str(self.cluster_list)

    def merge_clusters(self, cluster_b):
        self.cluster_list.extend(cluster_b.cluster_list)
        if self.cluster_label > cluster_b.cluster_label:
            self.cluster_label = cluster_b.cluster_label

    def get_distance(self, data_point_a, data_point_b):
        return np.linalg.norm(data_point_b - data_point_a)

    def get_linkage(self, cluster_b):
        if self.linkage == 'single':
            return self.single_linkage(cluster_b)
        elif self.linkage == 'complete':
            return self.complete_linkage(cluster_b)
        elif self.linkage == 'centroid':
            return self.centroid_linkage(cluster_b)
        else:
            raise ValueError("method not defined")

    def centroid_linkage(self, cluster_b):
        centroid_a = np.mean(self.cluster_list, axis=0)
        centroid_b = np.mean(cluster_b.cluster_list, axis=0)
        return np.linalg.norm(centroid_a - centroid_b)

    def single_linkage(self, cluster_b):
        min_distance = float('inf')
        for data_point_a in self.cluster_list:
            for data_point_b in cluster_b.cluster_list:
                distance = self.get_distance(data_point_a, data_point_b)
                if distance < min_distance:
                    min_distance = distance
        return min_distance

    def complete_linkage(self, cluster_b):
        max_distance = float('-inf')
        for data_point_a in self.cluster_list:
            for data_point_b in cluster_b.cluster_list:
                distance = self.get_distance(data_point_a, data_point_b)
                if distance > max_distance:
                    max_distance = distance
        return max_distance

class AgglomerativeMethod:
    def __init__(self, data=None, cluster_number=None, linkage_method='centroid'):
        self.cluster_number = cluster_number
        self.linkage_method = linkage_method
        self.data = data
        self.labels = []
        self.cluster_list = {}
        self.heap_max_size = len(data)
        self.distance_matrix = np.zeros((len(data), len(data)))
        self.distance_triple = []
        self.heap_cluster = None

    def init_clusters(self):
        for i, data_point in enumerate(self.data):
            new_cluster = Cluster(cluster_label=i, data_point=data_point, linkage=self.linkage_method)
            self.cluster_list[i] = new_cluster

    def init_distance_matrix_heap(self):
        self.distance_matrix = squareform(pdist(self.data, 'euclidean'))

        for label_a, clusterA in self.cluster_list.items():
            for label_b, clusterB in self.cluster_list.items():
                if label_a < label_b:
                    distance = self.distance_matrix[label_a][label_b]
                    triplet = (distance, label_a, label_b)
                    self.distance_triple.append(triplet)

        self.heap_cluster = HeapCluster(self.distance_triple)
        self.heap_cluster.build_heap()

    def get_min_distance(self, label_a, label_b):
        while True:
            result = self.heap_cluster.pop()
            if result == (float('inf'), None, None):
                return result
            if result[1] in (label_a, label_b) or result[2] in (label_a, label_b):
                continue
            if result[1] not in self.cluster_list or result[2] not in self.cluster_list:
                continue
            return result

    def min_heap_add_distances_new_cluster(self, new_cluster):
        label_a = new_cluster.cluster_label
        for cluster in self.cluster_list.values():
            label_b = cluster.cluster_label
            if label_a != label_b:
                distance = new_cluster.get_linkage(cluster)
                if label_a < label_b:
                    new_tuple = (distance, label_a, label_b)
                else:
                    new_tuple = (distance, label_b, label_a)
                self.heap_cluster.insert_k_mins(new_tuple, self.heap_max_size)

    def fit(self):
        self.init_clusters()
        self.init_distance_matrix_heap()

        while len(self.cluster_list) > self.cluster_number:
            print(len(self.cluster_list))
            min_value = self.heap_cluster.pop()
            cluster_a_index = min_value[1]
            cluster_b_index = min_value[2]

            if cluster_a_index not in self.cluster_list or cluster_b_index not in self.cluster_list:
                continue

            cluster_a = self.cluster_list[cluster_a_index]
            cluster_b = self.cluster_list[cluster_b_index]

            cluster_a.merge_clusters(cluster_b)
            del self.cluster_list[cluster_b_index]

            self.min_heap_add_distances_new_cluster(cluster_a)

        self.assign_labels()

    def assign_labels(self):
        labels = np.zeros(len(self.data), dtype=int)
        for cluster_label, cluster_obj in self.cluster_list.items():
            for data_point in cluster_obj.cluster_list:
                index = np.where((self.data == data_point).all(axis=1))[0][0]
                labels[index] = cluster_label
        self.labels = labels

# Entrenar el modelo con los datos proporcionados y evaluar los resultados
if __name__ == "__main__":
    # Número de clusters deseados
    n_clusters = len(np.unique(true_labels))

    # Crear el modelo de Agglomerative Clustering
    clustering = AgglomerativeMethod(data=data_mean, cluster_number=n_clusters, linkage_method='centroid')

    # Ajustar el modelo y predecir las etiquetas
    clustering.fit()
    predicted_labels = clustering.labels

    # Evaluar el rendimiento
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)

    print("Adjusted Rand Index:", ari)
    print("Mutual Info Index:", nmi)
