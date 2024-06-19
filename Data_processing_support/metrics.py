def metricas(data,true_labels,pred_labels,type_reduce = ''):
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    from sklearn.metrics.cluster import adjusted_rand_score ,adjusted_mutual_info_score
    s_core = silhouette_score(data,pred_labels)
    ari = adjusted_rand_score(true_labels, pred_labels)
    ami = adjusted_mutual_info_score(true_labels, pred_labels)
    print('----------------------------------------')
    print(f"Silhouette Score - data {type_reduce}: {s_core}")
    print('----------------------------------------')
    print(f"Adjusted Rand Index - data {type_reduce}: {ari}")
    print('----------------------------------------')
    print(f"Mutual Info Index - data {type_reduce}: {ami}")
    print('----------------------------------------')
    print('\n')