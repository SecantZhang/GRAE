import unittest
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from datetime import datetime

from grae.models.grvae_models import GRVAE
from grae.models.siamese_ae import *
from grae.metrics import dtw_distance
from grae.data.timeseries_dataset import TSData

import sys
sys.path.append("/home/zzt7020/NUDB/deepsketch-clean")
# formatted_date = datetime.now().strftime("%y%m%d")
formatted_date = "250207"

class GRAETest(unittest.TestCase):
    train_data = None
    valid_data = None
    train_dataset = None
    valid_dataset = None
    train_data_large_path = "/home/zzt7020/NUDB/clones/GRAE/data/train_1k.npy"
    valid_data_large_path = "/home/zzt7020/NUDB/clones/GRAE/data/valid_large.npy"
    random_state = 1024

    lam = 100
    epochs = 300
    n_components = 8
    lr = .001
    batch_size = 128

    @classmethod
    def setUpClass(cls):
        cls.train_data = np.load(cls.train_data_large_path)
        cls.valid_data = np.load(cls.valid_data_large_path)
        cls.train_dataset = TSData(cls.train_data, cls.train_data, "none", 1, cls.random_state, augment=False)
        cls.valid_dataset = TSData(cls.valid_data, cls.valid_data, "none", 1, cls.random_state, augment=False)

    def test_GRVAE_normal_DTW(self):
        train_data_small, _ = train_test_split(self.train_data, train_size=1000, random_state=self.random_state)
        train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)

        model = GRVAE(dist_func="dtw", dist_loss_type="distdistributionloss", epochs=self.epochs, n_components=self.n_components, lr=0.0001,
                      batch_size=self.batch_size)
        model.fit(train_dataset_small)
        orig_dist = model.orig_dists.detach().cpu().numpy()
        emb_dist = model.emb_dists.detach().cpu().numpy()
        np.savetxt(f"/home/zzt7020/NUDB/clones/GRAE/tests/saved/logs/{formatted_date}/orig_dists-distdistributionloss-dtw.csv", orig_dist, delimiter=",", fmt="%.4f")
        np.savetxt(f"/home/zzt7020/NUDB/clones/GRAE/tests/saved/logs/{formatted_date}/emb_dists-distdistributionloss-dtw.csv", emb_dist, delimiter=",", fmt="%.4f")

    def test_GRVAE_normal_SinkHornLoss_DTW(self):
        train_data_small, _ = train_test_split(self.train_data, train_size=1000, random_state=self.random_state)
        train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)

        model = GRVAE(dist_func="dtw", dist_loss_type="sinkhornloss", epochs=self.epochs, n_components=self.n_components, lr=0.0001,
                      batch_size=self.batch_size)
        model.fit(train_dataset_small)
        orig_dist = model.orig_dists.detach().cpu().numpy()
        emb_dist = model.emb_dists.detach().cpu().numpy()
        np.savetxt(f"/home/zzt7020/NUDB/clones/GRAE/tests/saved/logs/{formatted_date}/orig_dists-SinkHornLoss-dtw.csv", orig_dist, delimiter=",", fmt="%.4f")
        np.savetxt(f"/home/zzt7020/NUDB/clones/GRAE/tests/saved/logs/{formatted_date}/emb_dists-SinkHornLoss-dtw.csv", emb_dist, delimiter=",", fmt="%.4f")

    def test_GRVAE_normal_DistanceDiffLoss_DTW(self):
        train_data_small, _ = train_test_split(self.train_data, train_size=1000, random_state=self.random_state)
        train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)

        model = GRVAE(dist_func="dtw", dist_loss_type="distancediffloss", epochs=self.epochs, n_components=self.n_components, lr=0.0001,
                      batch_size=self.batch_size)
        model.fit(train_dataset_small)
        orig_dist = model.orig_dists.detach().cpu().numpy()
        emb_dist = model.emb_dists.detach().cpu().numpy()
        np.savetxt(f"/home/zzt7020/NUDB/clones/GRAE/tests/saved/logs/{formatted_date}/orig_dists-distancediffloss-dtw.csv", orig_dist, delimiter=",", fmt="%.4f")
        np.savetxt(f"/home/zzt7020/NUDB/clones/GRAE/tests/saved/logs/{formatted_date}/emb_dists-distancediffloss-dtw.csv", emb_dist, delimiter=",", fmt="%.4f")

    def test_GRVAE_normal_Qetch(self):
        train_data_small, _ = train_test_split(self.train_data, train_size=1000, random_state=self.random_state)
        train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)

        model = GRVAE(dist_func="qetch", dist_loss_type="distdistributionloss", epochs=100, n_components=self.n_components, lr=0.0001,
                      batch_size=self.batch_size)
        model.fit(train_dataset_small)
        orig_dist = model.orig_dists.detach().cpu().numpy()
        emb_dist = model.emb_dists.detach().cpu().numpy()
        np.savetxt(f"/home/zzt7020/NUDB/clones/GRAE/tests/saved/logs/{formatted_date}/orig_dists-distdistributionloss-qet.csv", orig_dist, delimiter=",", fmt="%.4f")
        np.savetxt(f"/home/zzt7020/NUDB/clones/GRAE/tests/saved/logs/{formatted_date}/emb_dists-distdistributionloss-qet.csv", emb_dist, delimiter=",", fmt="%.4f")

    def test_GRVAE_normal_SinkHornLoss_Qetch(self):
        train_data_small, _ = train_test_split(self.train_data, train_size=1000, random_state=self.random_state)
        train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)

        model = GRVAE(dist_func="qetch", dist_loss_type="sinkhornloss",epochs=100, n_components=self.n_components, lr=0.0001,
                      batch_size=self.batch_size)
        model.fit(train_dataset_small)
        orig_dist = model.orig_dists.detach().cpu().numpy()
        emb_dist = model.emb_dists.detach().cpu().numpy()
        np.savetxt(f"/home/zzt7020/NUDB/clones/GRAE/tests/saved/logs/{formatted_date}/orig_dists-sinkhornloss-qet.csv", orig_dist, delimiter=",", fmt="%.4f")
        np.savetxt(f"/home/zzt7020/NUDB/clones/GRAE/tests/saved/logs/{formatted_date}/emb_dists-sinkhornloss-qet.csv", emb_dist, delimiter=",", fmt="%.4f")

    def test_GRVAE_normal_DistanceDiffLoss_Qetch(self):
        train_data_small, _ = train_test_split(self.train_data, train_size=1000, random_state=self.random_state)
        train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)

        model = GRVAE(dist_func="qetch", dist_loss_type="distancediffloss", epochs=100, n_components=self.n_components, lr=0.0001,
                      batch_size=self.batch_size)
        model.fit(train_dataset_small)
        orig_dist = model.orig_dists.detach().cpu().numpy()
        emb_dist = model.emb_dists.detach().cpu().numpy()
        np.savetxt(f"/home/zzt7020/NUDB/clones/GRAE/tests/saved/logs/{formatted_date}/orig_dists-distancediffloss-qet.csv", orig_dist, delimiter=",", fmt="%.4f")
        np.savetxt(f"/home/zzt7020/NUDB/clones/GRAE/tests/saved/logs/{formatted_date}/emb_dists-distancediffloss-qet.csv", emb_dist, delimiter=",", fmt="%.4f")