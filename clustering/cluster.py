
import numpy as np
from tqdm import tqdm
import infomap
from .knn import *
from .utils import *


class ClusterInfomap:
    def __init__(self,
                knn_method = "faiss-cpu",
                min_sim = 0.8,
                k = 30
            ):

        self.knn_method = knn_method
        self.min_sim = min_sim
        self.k = k

    def get_links(self, single, links, nbrs, dists):
        for i in tqdm(range(nbrs.shape[0])):
            count = 0
            for j in range(0, len(nbrs[i])):
                if i == nbrs[i][j]:
                    pass
                elif dists[i][j] <= 1 - self.min_sim:
                    count += 1
                    links[(i, nbrs[i][j])] = float(1 - dists[i][j])
                else:
                    break
            if count == 0:
                single.append(i)
        return single, links

    def knns2ordered_nbrs(self, knns, sort=True):
        if isinstance(knns, list):
            knns = np.array(knns)
        nbrs = knns[:, 0, :].astype(np.int32)
        dists = knns[:, 1, :]
        if sort:
            # sort dists from low to high
            nb_idx = np.argsort(dists, axis=1)
            idxs = np.arange(nb_idx.shape[0]).reshape(-1, 1)
            dists = dists[idxs, nb_idx]
            nbrs = nbrs[idxs, nb_idx]
        return dists, nbrs


    def get_dist_nbr(self, features, k=80, knn_method='faiss-cpu'):
        features = l2norm(features)
        features = np.ascontiguousarray(features)        
        index = knn_faiss(feats=features, k=k, knn_method=knn_method)
        knns = index.get_knns()
        dists, nbrs = self.knns2ordered_nbrs(knns)
        return dists, nbrs


    def cluster_by_infomap(self, nbrs, dists):
        single = []
        links = {}
        single, links = self.get_links(single=single, links=links, nbrs=nbrs, dists=dists)
        infomapWrapper = infomap.Infomap("--two-level --directed")
        for (i, j), sim in tqdm(links.items()):
            _ = infomapWrapper.addLink(int(i), int(j), sim)
        infomapWrapper.run()

        label2idx = {}
        idx2label = {}
        for node in infomapWrapper.iterTree():
            # node.physicalId -> id of feature vector
            # node.moduleIndex() -> cluster id
            idx2label[node.physicalId] = node.moduleIndex()
            if node.moduleIndex() not in label2idx:
                label2idx[node.moduleIndex()] = []
            label2idx[node.moduleIndex()].append(node.physicalId)

        node_count = 0
        for k, v in label2idx.items():
            if k == 0:
                node_count += len(v[2:])
                label2idx[k] = v[2:]
            else:
                node_count += len(v[1:])
                label2idx[k] = v[1:]

        keys_len = len(list(label2idx.keys()))

        for single_node in single:
            idx2label[single_node] = keys_len
            label2idx[keys_len] = [single_node]
            keys_len += 1

        pred_labels = intdict2ndarray(idx2label)
        return pred_labels

    def do_clustering(self, features):
        dists, nbrs = self.get_dist_nbr(features, k = self.k, knn_method=self.knn_method)
        pred_labels = self.cluster_by_infomap(nbrs, dists)
        return pred_labels