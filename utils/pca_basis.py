# PCA label basis for TransDF, fit per channel over the time axis on the train split
import hashlib
import os

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def label_windows(dataset):
    # [N x pred_len x D] strided view of the training labels
    data_y = np.ascontiguousarray(dataset.data_y, dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(data_y, dataset.pred_len, axis=0)
    windows = windows[dataset.seq_len:dataset.seq_len + len(dataset)]
    return windows.transpose(0, 2, 1)


def get_pca_base(windows):
    N, T, D = windows.shape
    components, means, stds = [], [], []
    for d in range(D):
        scaler = StandardScaler()
        chunk = scaler.fit_transform(np.ascontiguousarray(windows[:, :, d]))
        pca = PCA(n_components=max(1, min(T, N)))
        pca.fit(chunk)
        components.append(np.asarray(pca.components_))
        means.append(scaler.mean_)
        stds.append(scaler.scale_)
    base = np.array(components)                                              # [D x rank x T]
    initializer = np.array([np.array(means).T, np.array(stds).T])            # [2 x T x D]
    return base.astype(np.float32), initializer.astype(np.float32)


def fit_or_load_pca_base(train_data, args, cache_dir='./pca_cache'):
    key = '{}_{}_ft{}_sl{}_pl{}'.format(args.data, os.path.splitext(args.data_path)[0], args.features,
                                        args.seq_len, args.pred_len)
    path = os.path.join(cache_dir, '{}_{}.npz'.format(key, hashlib.md5(key.encode()).hexdigest()[:8]))
    if os.path.exists(path):
        cached = np.load(path)
        return cached['components'], cached['initializer']
    components, initializer = get_pca_base(label_windows(train_data))
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, components=components, initializer=initializer)
    return components, initializer


class BasisCache:
    def __init__(self, components, initializer):
        self._components = torch.as_tensor(np.asarray(components)).float()
        self._initializer = torch.as_tensor(np.asarray(initializer)).float()
        self._cache = {}

    def to(self, device):
        if device not in self._cache:
            self._cache[device] = (self._components.to(device), self._initializer.to(device))
        return self._cache[device]


def pca_torch(data, pca_cache):                                              # data: [B x T x D]
    components, initializer = pca_cache.to(data.device)
    data = (data - initializer[0]) / initializer[1]
    return torch.einsum('btd,rt->brd', data, components.mean(dim=0))
