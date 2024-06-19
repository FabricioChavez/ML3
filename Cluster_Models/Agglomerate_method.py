import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix, KDTree
from scipy.spatial.distance import pdist, squareform
from heap import HeapCluster


class cluster:

    def __init__(self, cluster_label=None, data_point=None, linkage='centroid'):
        """
          cluster_label --> category assigned to the cluster
          data_point --> data point to be clustered
         """
        self.cluster_label = cluster_label
        self.data_point = data_point
        self.cluster_list = []
        self.cluster_list.append(self.data_point)
        self.linkage = linkage

    def __str__(self):
        return "Current cluster is " + str(self.cluster_list)

    def merge_clusters(self, cluster_b):
        if self.cluster_label > cluster_b.cluster_label:
            self.cluster_label = cluster_b.cluster_label
        self.cluster_list.extend(cluster_b.cluster_list)

    def get_distance(self, data_point_a, data_point_b):
        return np.linalg.norm(data_point_b - data_point_a)

    def get_linkage(self, cluster_b):
        if self.linkage == 'single':
            return self.single_linkage(cluster_b)
        elif self.linkage == 'complete':
            return self.complete_linkage(cluster_b)
        elif self.linkage == 'centroid':
            return self.centroid_list(cluster_b)
        else:
            print("method not defined")
            exit(1)

    def centroid_list(self, cluster_b):
        centroid_a = np.mean(self.cluster_list, axis=0)
        centroid_b = np.mean(cluster_b.cluster_list, axis=0)

        return np.linalg.norm(centroid_a - centroid_b)

    def single_linkage(self, cluster_b):
        cluster_a_tree = KDTree(self.cluster_list)
        cluster_b_tree = KDTree(cluster_b.cluster_list)

        min_distance = float('inf')

        for data_point_a in self.cluster_list:
            distance, _ = cluster_a_tree.query(data_point_a)
            min_distance = min(min_distance, distance)

        for data_point_b in self.cluster_list:
            distance, _ = cluster_b_tree.query(data_point_b)
            min_distance = min(min_distance, distance)

        return min_distance

    def complete_linkage(self, cluster_b):
        max_distance = float('-inf')
        for data_point_a in self.cluster_list:
            for data_point_b in self.cluster_list:
                distance, _ = self.get_distance(data_point_a, data_point_b)
                if distance > max_distance:
                    max_distance = distance

        return max_distance


class AgglomerativeMethod:

    def __init__(self, data=None, cluster_number=None, linkage_method=None):
        """
            cluster_number --> number of clusters to use
            linkage_method --> linkage method to use the ones available are {simple_linkage , absolute_linkage}
            distance are calculated using Euclidean distance
        """
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
        for i, data_point in zip(range(self.data.shape[0]), self.data):
            new_cluster = cluster(cluster_label=i, data_point=data_point)
            self.cluster_list[i] = new_cluster

    def init_distance_matrix_heap(self):
        print(len(self.cluster_list))
        self.distance_matrix = squareform(pdist(self.data, 'euclidean'))
        print("WE have distance matrix")

        for label_a, clusterA in self.cluster_list.items():
            mini_tuple = (float('inf'), None, None)

            for label_b, clusterB in self.cluster_list.items():
                if label_a < label_b:
                    distance = self.distance_matrix[label_a][label_b]
                    triplet = (distance, label_a, label_b)
                    if triplet[0] < mini_tuple[0]:
                        mini_tuple = triplet

            self.distance_triple.append(mini_tuple)

        self.heap_cluster = HeapCluster(self.distance_triple)
        print("Process ended for distance_matrix")
        self.heap_max_size.build_heap()
        print("Process ended for heap build")
        print(self.heap_max_size)
        print(len(self.heap_cluster.heap))

    def get_min_distance(self, label_a, label_b):

        result = self.heap_cluster.pop()

        while result[1] == label_a or result[1] == label_b:
            result = self.heap_cluster.pop()

        return result

    def min_heap_add_distances_new_cluster(self, new_cluster):
        label_a = new_cluster.cluster_label
        for cluster in self.cluster_list.values():
            label_b = cluster.cluster_label
            if label_a != label_b:
                distance = new_cluster.get_linkage(cluster)
                if label_a < label_b:
                    new_tuple = (distance, label_a, label_b)
                    self.heap_cluster.insert_k_mins(new_tuple, self.heap_max_size)
                else:
                    new_tuple = (distance, label_b, label_a)
                    self.heap_cluster.insert_k_mins(new_tuple, self.heap_max_size)

                self.heap_cluster.insert(new_tuple)

    def fit(self):
        self.init_clusters()
        self.init_distance_matrix_heap()
        print("CLUSTER LIST")

        print("EMPIEZA LO WENO :3 uwu UOHHHHHH ToT")
        min_value = self.heap_max_size.pop()
        while len(self.cluster_list) > self.cluster_number or len(self.cluster_list) > 1:
            print(self.cluster_list)
            cluster_a_index = min_value[1]  #get first coord
            cluster_b_index = min_value[2]  #get second coord
            cluster_a = self.cluster_list[cluster_a_index]  #get cluster by index using dict
            cluster_b = self.cluster_list[cluster_b_index]  #get cluster by index using dict
            print("cluster to merge", cluster_a.cluster_label, cluster_b.cluster_label)
            cluster_a.merge_clusters(cluster_b)
            print("new cluster -->", cluster_a.cluster_label)
            del self.cluster_list[cluster_a_index]
            del self.cluster_list[cluster_b_index]
            self.cluster_list[cluster_a.cluster_label] = cluster_a
            min_value = self.get_min_distance(cluster_a_index, cluster_b_index)
            self.min_heap_add_distances_new_cluster(cluster_a)
            min_value = min()
            print("new distances to cluster added")

    def fit_a(self):

        print("EMPIEZA LO WENO :3 uwu UOHHHHHH ToT")
        while len(self.cluster_list) > self.cluster_number or len(self.cluster_list) > 1:
            min_value = self.get_min_distance()

            cluster_a_index = min_value[1]  #get first coord
            cluster_b_index = min_value[2]  #get second coord
            cluster_a = self.cluster_list[cluster_a_index]  #get cluster by index using dict
            cluster_b = self.cluster_list[cluster_b_index]  #get cluster by index using dict
            print("cluster to merge", cluster_a.cluster_label, cluster_b.cluster_label)
            cluster_a.merge_clusters(cluster_b)
            # print("new cluster -->", cluster_a.cluster_label)
            del self.cluster_list[cluster_a_index]
            del self.cluster_list[cluster_b_index]
            self.cluster_list[cluster_a.cluster_label] = cluster_a
            self.min_heap_add_distances_new_cluster(cluster_a)
            print("new distances to cluster added")


if __name__ == '__main__':
    #    np_mean = pd.read_csv('train_mean.csv')

    data = np.array([[1, 2], [2, 3], [3, 4], [8, 9], [9, 10], [10, 11]])

    #   np_mean = np_mean.to_numpy()
    cluster_method = AgglomerativeMethod(data=data, cluster_number=2)
    cluster_method.fit()
