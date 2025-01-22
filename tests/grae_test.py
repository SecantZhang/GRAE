import unittest
import numpy as np
import torch
from sklearn.model_selection import train_test_split

from grae.models.grae_models import GRAEUMAP_Strict
from grae.models.siamese_ae import *
from grae.metrics import dtw_distance
from grae.data.timeseries_dataset import TSData

import sys
sys.path.append("/home/zzt7020/NUDB/deepsketch-clean")


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
    batch_size = 256

    @classmethod
    def setUpClass(cls):
        cls.train_data = np.load(cls.train_data_large_path)
        cls.valid_data = np.load(cls.valid_data_large_path)
        cls.train_dataset = TSData(cls.train_data, cls.train_data, "none", 1, cls.random_state, augment=False)
        cls.valid_dataset = TSData(cls.valid_data, cls.valid_data, "none", 1, cls.random_state, augment=False)

    def testGRAEUmapTrainSmallDTW(self):
        train_data_small, _ = train_test_split(self.train_data, train_size=1000, random_state=self.random_state)
        train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)
        model = GRAEUMAP_Strict(lam=self.lam, epochs=self.epochs, n_components=self.n_components, lr=self.lr,
                                batch_size=self.batch_size, metric=dtw_distance, output_metric="euclidean", verbose=True)
        model.fit(train_dataset_small)

    def testGRAEUmapLargeDTW(self):
        model = GRAEUMAP_Strict(lam=self.lam, epochs=self.epochs, n_components=self.n_components, lr=self.lr,
                                batch_size=self.batch_size, metric=dtw_distance, output_metric="euclidean", verbose=True)

    def testSiameseAE_DTW_Small(self):
        train_data_small, _ = train_test_split(self.train_data, train_size=2000, random_state=self.random_state)
        train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)
        model = SiameseAE_DTW(epochs=self.epochs, n_components=self.n_components, lr=self.lr,
                              batch_size=self.batch_size)
        model.fit(train_dataset_small)

    def testSiameseAE_DTW_Small_RankedLoss(self):
        train_data_small, _ = train_test_split(self.train_data, train_size=2000, random_state=self.random_state)
        train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)
        model = SiameseAE_RankedLoss_DTW(epochs=self.epochs, n_components=self.n_components, lr=0.001,
                                         batch_size=self.batch_size)
        model.fit(train_dataset_small)
        model.save("/home/zzt7020/NUDB/clones/GRAE/tests/saved/models/testSiameseAEDTW_RankedLoss_Small_lr001.pth")

    def testSiameseAE_DTW_Small_MRRLoss(self):
        train_data_small, _ = train_test_split(self.train_data, train_size=2000, random_state=self.random_state)
        train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)
        model = SiameseAE_MRRLoss_DTW(epochs=self.epochs, n_components=self.n_components, lr=0.001,
                                      batch_size=self.batch_size)
        model.fit(train_dataset_small)
        model.save("/home/zzt7020/NUDB/clones/GRAE/tests/saved/models/testSiameseAEDTW_MRRLoss_Small_lr001.pth")

    def testSiameseAE_Qetch_Small_lr001(self):
        train_data_small, _ = train_test_split(self.train_data, train_size=2000, random_state=self.random_state)
        train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)
        model = SiameseAE_Qetch(epochs=self.epochs, n_components=self.n_components, lr=.001,
                                batch_size=self.batch_size, num_reports=self.epochs)
        model.fit(train_dataset_small)
        model.save("/home/zzt7020/NUDB/clones/GRAE/tests/saved/models/testSiameseAEQetchSmall_lr001.pth")

    def testSiameseAEQetchSmall_lr0005(self):
        train_data_small, _ = train_test_split(self.train_data, train_size=500, random_state=self.random_state)
        train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)
        model = SiameseAE_Qetch(epochs=self.epochs, n_components=self.n_components, lr=.0005,
                                batch_size=self.batch_size, num_reports=self.epochs, loss_weight=[0, 0, 1])
        model.fit(train_dataset_small)
        model.save("/home/zzt7020/NUDB/clones/GRAE/tests/saved/models/testSiameseAEQetchSmall_lr0005.pth")

    def testSiameseAEQetchSmall_RankedLoss_lr0005(self):
        train_data_small, _ = train_test_split(self.train_data, train_size=2000, random_state=self.random_state)
        train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)
        model = SiameseAE_RankedLoss_Qetch(epochs=self.epochs, n_components=self.n_components, lr=.0005,
                                           batch_size=self.batch_size, num_reports=self.epochs)
        model.fit(train_dataset_small)
        model.save("/home/zzt7020/NUDB/clones/GRAE/tests/saved/models/testSiameseAEQetch_RankedLoss_Small_lr0005.pth")

    # def testSiameseAEQetchSmallVAE_lr0005(self):
    #     train_data_small, _ = train_test_split(self.train_data, train_size=2000, random_state=self.random_state)
    #     train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)
    #     model = SiameseAE_Qetch(epochs=self.epochs, n_components=self.n_components, lr=.0005,
    #                             batch_size=self.batch_size, num_reports=self.epochs, vae=True)
    #     model.fit(train_dataset_small)
    #     model.save("/home/zzt7020/NUDB/clones/GRAE/tests/saved/models/testSiameseAEQetchSmallVAE_lr0005.pth")

    def testSiameseAEQetchSmall_lr0005_output32(self):
        train_data_small, _ = train_test_split(self.train_data, train_size=2000, random_state=self.random_state)
        train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)
        model = SiameseAE_Qetch(epochs=self.epochs, n_components=32, lr=.0005,
                                batch_size=self.batch_size, num_reports=self.epochs)
        model.fit(train_dataset_small)
        model.save("/home/zzt7020/NUDB/clones/GRAE/tests/saved/models/testSiameseAEQetchSmall_lr0005_output32.pth")

    def testSiameseAEQetchSmall_MRRLoss_lr0005(self):
        train_data_small, _ = train_test_split(self.train_data, train_size=2000, random_state=self.random_state)
        train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)
        model = SiameseAE_MRRLoss_Qetch(epochs=self.epochs, n_components=self.n_components, lr=.0005,
                                        batch_size=self.batch_size, num_reports=self.epochs)
        model.fit(train_dataset_small)
        model.save("/home/zzt7020/NUDB/clones/GRAE/tests/saved/models/testSiameseAEQetch_mrrloss_Small_lr0005.pth")

    def testSiameseAE_Qetch_Small_MRRLoss_lr0005_deep(self):
        train_data_small, _ = train_test_split(self.train_data, train_size=2000, random_state=self.random_state)
        train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)
        model = SiameseAE_RankedLoss_Qetch_DeepLayers(epochs=self.epochs, n_components=self.n_components, lr=.0005,
                                        batch_size=self.batch_size, num_reports=self.epochs)
        model.fit(train_dataset_small)
        model.save("/home/zzt7020/NUDB/clones/GRAE/tests/saved/models/testSiameseAEQetch_mrrloss_Small_lr0005.pth")

    def testSiameseAE_Qetch_Small_corrloss_lr0005_deep(self):
        train_data_small, _ = train_test_split(self.train_data, train_size=2000, random_state=self.random_state)
        train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)
        model = SiameseAE_CorrLoss_Qetch_DeepLayers(epochs=self.epochs, n_components=self.n_components, lr=.0005,
                                                    batch_size=self.batch_size, num_reports=self.epochs, eval=False)
        model.fit(train_dataset_small)
        model.save("/home/zzt7020/NUDB/clones/GRAE/tests/saved/models/testSiameseAEQetch_corrloss_Small_lr0005_deeplayers.pth")

    def test_siamese_ae_sbd_small_base(self):
        train_data_small, _ = train_test_split(self.train_data, train_size=2000, random_state=self.random_state)
        train_dataset_small = TSData(train_data_small, train_data_small, "none", 1, self.random_state, augment=False)
        model = SiameseAE_SBD(epochs=self.epochs, n_components=self.n_components, lr=.0005,
                              batch_size=self.batch_size, num_reports=self.epochs)
        model.fit(train_dataset_small)
        model.save("/home/zzt7020/NUDB/clones/GRAE/tests/saved/models/testSiameseAE_sbd_lr0005.pth")

