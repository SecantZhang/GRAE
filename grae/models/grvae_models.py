import os

import torch
import torch.nn as nn
import numpy as np
import scipy
import tqdm
import math
import sys

from grae.models.grae_models import AE, GRAEBase
from grae.models.losses import LatentCovLoss, DistDistributionLoss, SinkhornLoss, DistanceDiffLoss
from grae.metrics.soft_dtw_cuda import SoftDTW
from grae.metrics.qetch_torch_new import qetch_batched
from grae.metrics.sbd_torch import sbd_1d as sbd_torch
from grae.utils.similarity_utils import conv_numerical_to_ranks_tensor, metrics_store, metric_PN_at_K, pairwise_distance_matrix
from typing import Callable, Union
from geomloss import SamplesLoss


class GRVAEBase(AE):
    def __init__(self, *,
                 dist_func: Union[str, Callable],
                 embedding_dist_func: Union[str, Callable] = "euclidean",
                 loss_weight=None,
                 num_reports: int = 200,
                 dist_loss_type: str = "distdistributionloss",
                 eval: bool = False,
                 **kwargs):
        """

        Args:
            dist_func:
            **kwargs:
        """
        super().__init__(**kwargs)
        supported_loss = ["distdistributionloss", "sinkhornloss", "distancediffloss"]
        if loss_weight is None:
            self.loss_weight = [0, 0, 1]
        else:
            assert len(loss_weight) == 3
            self.loss_weight = loss_weight

        if dist_loss_type not in supported_loss:
            raise NotImplementedError(f"loss type {dist_loss_type} not supported yet, please refer to {supported_loss}")

        self.dist_func = metrics_store(dist_func, data_type="torch")
        self.emb_dist_func = metrics_store(embedding_dist_func, data_type="torch")
        self.num_reports = num_reports
        self.loss_report_step = math.ceil(self.epochs / self.num_reports)
        self.dist_loss_type = dist_loss_type
        self.batch_loss_list = []
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.eval = eval
        self.orig_dists: torch.Tensor = None
        self.emb_dists: torch.Tensor = None

        # normalization
        self.batch_norm = nn.BatchNorm1d(self.n_components)

        self.original_data_dist_ind = None

        print(f"Configuration: \n"
              f"dist_func      = {self.dist_func}, \n"
              f"emb dist_func  = {self.emb_dist_func}, \n"
              f"num_reports    = {num_reports}, \n"
              f"dist_loss_type = {dist_loss_type}")


    def fit(self, x):
        torch.manual_seed(self.random_state)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.data_shape = x[0][0].shape
        if self.torch_module is None:
            self.init_torch_module(self.data_shape, batch_norm=False)

        self.optimizer = torch.optim.Adam(self.torch_module.parameters(),
                                          lr=self.lr,
                                          weight_decay=self.weight_decay)
        # Train AE
        # Training steps are decomposed as calls to specific methods that can be overriden by children class if need be
        self.torch_module.train()

        self.loader = self.get_loader(x)

        # collecting dist data
        data_len = self.loader.dataset.data.shape[0]
        self.orig_dists = torch.zeros((self.epochs, data_len), device=self.DEVICE)
        self.emb_dists = torch.zeros((self.epochs, data_len), device=self.DEVICE)

        if self.data_val is not None:
            self.val_loader = self.get_loader(self.data_val)

        # Get first metrics
        self.log_metrics(0)


        # pbar = tqdm.tqdm(total=len(list(range(1, self.epochs + 1))), desc="fitting")
        for epoch in range(1, self.epochs + 1):
            # print(f'            Epoch {epoch}...')
            batch_losses = torch.tensor([], dtype=torch.float32, device=torch.device("cuda"))
            recon_losses = torch.tensor([], dtype=torch.float32, device=torch.device("cuda"))
            cov_losses = torch.tensor([], dtype=torch.float32, device=torch.device("cuda"))
            dist_losses = torch.tensor([], dtype=torch.float32, device=torch.device("cuda"))
            for batch_idx, batch in enumerate(self.loader):
                self.optimizer.zero_grad()
                batch_loss, (recon_loss, cov_loss, dist_loss) = self.train_body(batch, batch_idx, epoch)
                batch_losses = torch.cat((batch_losses, batch_loss.unsqueeze(0)))
                recon_losses = torch.cat((recon_losses, recon_loss.unsqueeze(0)))
                cov_losses   = torch.cat((cov_losses, cov_loss.unsqueeze(0)))
                dist_losses  = torch.cat((dist_losses, dist_loss.unsqueeze(0)))

                self.optimizer.step()
                # pbar.set_description(f"fitting, curr batch loss: {round(batch_loss.item(), 3):<6}")

            if epoch % self.loss_report_step == 0:
                print(f"epoch {epoch}/{self.epochs}: avg loss = {round(np.mean(batch_losses.tolist()), 3):<6}, "
                      f"recon loss = {round(self.loss_weight[0] * np.mean(recon_losses.tolist()), 3):<6}, "
                      f"cov loss = {round(self.loss_weight[1] * np.mean(cov_losses.tolist()), 3):<6}, "
                      f"dist loss = {round(self.loss_weight[2] *np.mean(dist_losses.tolist()), 3):<6}", flush=True)

            self.log_metrics(epoch)
            self.end_epoch(epoch)

            # Early stopping
            if self.early_stopping_count == self.patience:
                if self.comet_exp is not None:
                    self.comet_exp.log_metric('early_stopped',
                                              epoch - self.early_stopping_count)
                # pbar.set_description("fitting: early stopping triggered")
                break
            # pbar.update(1)
        # pbar.close()

        # Load checkpoint if it exists
        checkpoint_path = os.path.join(self.write_path, 'checkpoint.pt')

        if os.path.exists(checkpoint_path):
            self.load(checkpoint_path)
            os.remove(checkpoint_path)

    def compute_loss(self, x, x_hat, z, idx, batch_idx, epoch_idx):
        reconstruction_loss = self.criterion(x_hat, x)

        covariance_loss_func = LatentCovLoss()
        covariance_loss = covariance_loss_func(z)

        reshuffled_x_indices = torch.randperm(x.shape[0])
        x_reshuffled = x[reshuffled_x_indices]
        z_reshuffled = z[reshuffled_x_indices]
        dist_x = self.dist_func(x, x_reshuffled).to(torch.device("cuda"))
        dist_z = self.dist_func(z, z_reshuffled).to(torch.device("cuda"))  # TODO: should I use this?
        # dist_z = self.emb_dist_func(z, z_reshuffled).to(torch.device("cuda")) # TODO: should I use this?
        # TODO: Fix the gradient vanishing problem

        # collecting dist statistics
        self.orig_dists[epoch_idx-1, batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size] = dist_x
        self.emb_dists[epoch_idx-1, batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size] = dist_z

        if self.dist_loss_type == "distdistributionloss":
            corr_loss_func = DistDistributionLoss()
        elif self.dist_loss_type == "sinkhornloss":
            corr_loss_func = SinkhornLoss()
        elif self.dist_loss_type == "distancediffloss":
            corr_loss_func = DistanceDiffLoss()
        else:
            raise NotImplementedError()

        distance_loss = corr_loss_func(dist_x, dist_z)

        aggregated_loss = (self.loss_weight[0] * reconstruction_loss +
                           self.loss_weight[1] * covariance_loss +
                           self.loss_weight[2] * distance_loss)
        aggregated_loss.backward()

        return aggregated_loss, (reconstruction_loss, covariance_loss, distance_loss)

    def train_body(self, batch, batch_idx, epoch_idx):
        # same as the parent train_body
        data, _, idx = batch  # No need for labels. Training is unsupervised
        data = data.to(self.DEVICE)

        x_hat, z = self.torch_module(data)  # Forward pass

        # different: add batch norm layer to avoid gradient vanishing
        return self.compute_loss(data, x_hat, z, idx, batch_idx, epoch_idx)

    def end_epoch(self, epoch):
        if self.eval:
            # performs validation step to measure the quality of embeddings at each epoch.
            if epoch == 1:
                self.original_data_dist_ind = pairwise_distance_matrix(self.loader.dataset.data.to(self.DEVICE),
                                                                       metric=self.dist_func, return_type="indices")[:, :50]

            self.torch_module.eval()
            if epoch % 5 == 0:
                embedding_data = self.torch_module.encoder(self.loader.dataset.data.to(self.DEVICE))
                # reports the validation steps at the end of specific epochs
                pn_at_k = metric_PN_at_K(self.loader.dataset.data.to(self.DEVICE), embedding_data, 50, 100,
                                         original_data_dist_mat = self.original_data_dist_ind, metric=self.dist_func)
                print(f"pn_at_k at epoch {epoch} = {pn_at_k}")


class GRVAE(GRVAEBase):
    def __init__(self, dist_func: str, dist_loss_type: str = "distdistributionloss", **kwargs):
        super().__init__(dist_func=dist_func, dist_loss_type=dist_loss_type, **kwargs)