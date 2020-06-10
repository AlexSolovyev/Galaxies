import sklearn
import hdbscan

def start_clustering(x, y, params):
    hdbScan = hdbscan.hdbscan_.HDBSCAN().fit(x)

    rcsed_labels = hdbScan.labels_
    for i in range(len(rcsed_labels)):
        if rcsed_labels[i]==-1:
            rcsed_labels[i]=i+5000000

    true = y
    pred = rcsed_labels

    fms = round(sklearn.metrics.fowlkes_mallows_score(true, pred),5)
    ars = round(sklearn.metrics.adjusted_rand_score(true, pred),5)
    nmi = round(sklearn.metrics.normalized_mutual_info_score(true, pred),5)

    return fms, ars, nmi
