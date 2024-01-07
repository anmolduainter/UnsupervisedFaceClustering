import faiss
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import os

class knn():
    def __init__(self, feats, k, index_path='', verbose=True):
        pass

    def filter_by_th(self, i):
        th_nbrs = []
        th_dists = []
        nbrs, dists = self.knns[i]
        for n, dist in zip(nbrs, dists):
            if 1 - dist < self.th:
                continue
            th_nbrs.append(n)
            th_dists.append(dist)
        th_nbrs = np.array(th_nbrs)
        th_dists = np.array(th_dists)
        return th_nbrs, th_dists

    def get_knns(self, th=None):
        if th is None or th <= 0.:
            return self.knns
        nproc = 1
        self.th = th
        self.th_knns = []
        tot = len(self.knns)
        if nproc > 1:
            pool = mp.Pool(nproc)
            th_knns = list(
                tqdm(pool.imap(self.filter_by_th, range(tot)), total=tot))
            pool.close()
        else:
            th_knns = [self.filter_by_th(i) for i in range(tot)]
        return th_knns


class knn_faiss(knn):
    def __init__(self,
                 feats,
                 k,
                 index_path='',
                 knn_method='faiss-cpu',
                 verbose=True):

        knn_ofn = index_path + '.npz'
        if os.path.exists(knn_ofn):
            print('[{}] read knns from {}'.format(knn_method, knn_ofn))
            self.knns = np.load(knn_ofn)['data']
        else:
            feats = feats.astype('float32')
            size, dim = feats.shape
            index = faiss.IndexFlatIP(dim)
            index.add(feats)

        knn_ofn = index_path + '.npz'
        if os.path.exists(knn_ofn):
            pass
        else:
            sims, nbrs = index.search(feats, k=k)
            # torch.cuda.empty_cache()
            self.knns = [(np.array(nbr, dtype=np.int32),
                            1 - np.array(sim, dtype=np.float32))
                            for nbr, sim in zip(nbrs, sims)]
