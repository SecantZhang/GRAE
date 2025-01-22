import torch
import torch.nn as nn
import torch.nn.functional as F


def covariance(latent):
    """
    Compute the covariance matrix of the latent vectors
    """
    mean_latent = torch.mean(latent, dim=0, keepdim=True)
    centered_latent = latent - mean_latent
    cov_matrix = torch.matmul(centered_latent.t(), centered_latent) / (latent.size(0) - 1)
    return cov_matrix


class LatentCovLoss(nn.Module):
    """
    Latent covariance loss class
    Latent covariance weight is annealed by AnnealingCallback
    """
    def __init__(self, cov_weight=1.0):
        """
        :param cov_weight: initial covariance loss weight
        """
        super(LatentCovLoss, self).__init__()
        self.cov_weight = nn.Parameter(torch.tensor(cov_weight))

    def forward(self, latent):
        """
        :param latent: batch of latent vectors
        :return: weighted covariance loss
        """
        cov = torch.abs(covariance(latent))
        cov_square = cov * cov
        nbr_of_cov = latent.shape[-1] * (latent.shape[-1] - 1)
        cov_loss = (torch.sum(cov_square) - torch.trace(cov_square)) / float(nbr_of_cov)
        return self.cov_weight * cov_loss

    def extra_repr(self):
        return f'cov_weight={self.cov_weight.item()}'


class PearsonCorrLoss(nn.Module):
    def __init__(self):
        super(PearsonCorrLoss, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute the Pearson Correlation Loss.

        :param x: Predicted tensor (1D or 2D).
        :param y: Target tensor (same shape as x).
        :return: Loss based on negative Pearson correlation.
        """
        # Ensure inputs are 2D (batch_size, features)
        if x.ndim == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)

        # Compute means
        mean_x = torch.mean(x, dim=1, keepdim=True)
        mean_y = torch.mean(y, dim=1, keepdim=True)

        # Compute deviations
        x = x - mean_x
        y = y - mean_y

        # Compute the numerator (covariance)
        numerator = torch.sum(x * y, dim=1)

        # Compute the denominator (product of standard deviations)
        denominator = torch.sqrt(torch.sum(x ** 2, dim=1) * torch.sum(y ** 2, dim=1))

        # Pearson correlation
        corr = numerator / denominator

        # Loss is 1 - correlation (to minimize)
        loss = 1 - corr.mean()
        return loss


class SmoothedMeanReciprocalRankLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(SmoothedMeanReciprocalRankLoss, self).__init__()
        self.temperature = temperature

    def forward(self, original_distances, embedding_distances):
        rankings = torch.argsort(torch.argsort(original_distances, dim=-1), dim=-1) + 1
        smoothed_scores = F.softmax(-embedding_distances / self.temperature, dim=-1) # apply softmax for smoothness
        reciprocal_ranks = 1.0 / rankings.float()
        smoothed_mrr = (smoothed_scores * reciprocal_ranks).sum(dim=-1).mean()
        loss = 1 - smoothed_mrr
        return loss


class MRRLoss(nn.Module):
    def __init__(self, temperature=1.0, smoothed=True):
        """
        Initializes the Unified MRR Loss.
        Args:
            temperature (float): Smoothing parameter for the softmax function (used when smoothed=True).
            smoothed (bool): Whether to use the smoothed version of MRR.
        """
        super(MRRLoss, self).__init__()
        self.temperature = temperature
        self.smoothed = smoothed

    def forward(self, original_distances, embedding_distances):
        """
        Computes the MRR Loss.
        Args:
            original_distances (torch.Tensor): Tensor of shape (batch_size, num_candidates),
                                               where each row contains the original distances.
            embedding_distances (torch.Tensor): Tensor of shape (batch_size, num_candidates),
                                                where each row contains the predicted embedding distances.
        Returns:
            torch.Tensor: Scalar tensor representing the loss.
        """
        batch_size = original_distances.size(0)

        # Get ground truth rankings based on original distances
        ground_truth_ranks = torch.argsort(torch.argsort(original_distances)) + 1

        if self.smoothed:
            # Apply softmax with temperature to the embedding distances
            smoothed_scores = F.softmax(-embedding_distances / self.temperature, dim=0)
            reciprocal_ranks = 1.0 / ground_truth_ranks.float()  # Compute reciprocal rank
            mrr = (smoothed_scores * reciprocal_ranks).sum().mean() # Smoothed MRR
        else:
            # Predicted rank positions (ascending order of embedding distances)
            predicted_ranks = torch.argsort(torch.argsort(embedding_distances)) + 1
            ground_truth_rank_in_prediction = torch.gather(predicted_ranks, dim=0, index=ground_truth_ranks - 1) # Find the rank of the ground truth in predicted rankings
            reciprocal_ranks = 1.0 / ground_truth_rank_in_prediction.float()
            mrr = reciprocal_ranks.mean()

        return -mrr
