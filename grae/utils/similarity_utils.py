import numpy as np
import torch
import pprint

from scipy.stats import pearsonr, kendalltau, spearmanr
from typing import Union, Callable, Optional
from grae.metrics import qetch, qetch_torch_new, soft_dtw_cuda, dtw_distance
from scipy.spatial.distance import cdist
from tqdm import tqdm


def metrics_store(metric: Union[str, Callable] = "euclidean", data_type: str = "torch"):
    if isinstance(metric, Callable):
        return metric

    softdtw = soft_dtw_cuda.SoftDTW(use_cuda=torch.cuda.is_available())

    supported_metrics = {
        "euclidean": {
            "numpy": lambda a, b: np.linalg.norm(a-b),
            "torch": lambda a, b: (a - b).pow(2).sum(1).sqrt()
        },
        "dtw": {
            "numpy": dtw_distance,
            "torch": softdtw
        },
        "qetch": {
            "numpy": qetch,
            "torch": qetch_torch_new.qetch_batched
        }
    }

    return supported_metrics[metric][data_type]


def precision_metric(ground_truth: torch.Tensor, query_result: torch.Tensor, p: int = 1, k: int = 100):
    """
    Calculating the precision@k metrics - number of ground-truth results (p) appeared in k searched results
    Args:
        ground_truth: index of the ranked ground-truth distances
        query_result: index of the ranked query result distances
        p: number of ground truths to consider
        k: search result range

    Returns:
    """
    ground_truth_p = ground_truth[:p]
    query_result_k = query_result[:k]
    intersection = ground_truth_p[torch.isin(ground_truth_p, query_result_k)]
    return intersection.shape[0]


def eval_corr(x1, x2, mode=None, verbose=True): # x1, x2: (n_samples, n_features)
    """
    Evaluate the correlation between two sets of data.
    """
    supported_modes = ['pearson', 'kendalltau', 'spearman']
    if len(x1) != len(x2):
        raise ValueError('x1 and x2 must have the same length')
    if mode not in supported_modes:
        raise NotImplementedError(f"mode {mode} is not supported yet.")
    if mode == None:
        mode = supported_modes

    result = {}

    if 'pearson' in mode:
        result['pearson'] = pearsonr(x1, x2)
    if 'kendalltau' in mode:
        result['kendalltau'] = kendalltau(x1, x2)
    if 'spearman' in mode:
        result['spearman'] = spearmanr(x1, x2)

    if verbose:
        pprint.pprint(result)

    return result


def eval_knn(x1, x2, x1_dist_mat=None, x2_dist_mat=None, verbose=True):
    if x1_dist_mat == None:
        pass


def eval_embeddings(embeddings, original_data, mode=None, verbose=True,
                    embedding_metrics="euclidean", output_metrics="euclidean"):
    """ 
    Complete evaluation of the quality of embeddings
    """
    assert len(embeddings) == len(original_data) # making sure the length of the embeddings and the original data are the same
    if mode == "full": # performs full evaluation by creating nearest neighbor graphs
        pass
    elif mode == "reduced": # performs reduced evaluation by random shuffling
        reshuffled_indices = np.random.randperm(len(embeddings))
        # seen queries


def pairwise_dist_indices_calculation(x: Union[np.ndarray, torch.Tensor],
                                      y: Optional[Union[np.ndarray, torch.Tensor]] = None,
                                      k: int = 50,
                                      metric: Callable = 'euclidean') -> Union[np.ndarray, torch.Tensor]:
    # Ensure y is provided or defaults to x
    y = x if y is None else y

    # Determine if inputs are NumPy arrays or PyTorch tensors
    is_numpy = isinstance(x, np.ndarray)

    if not is_numpy:
        x, y = x.cpu().numpy(), y.cpu().numpy()  # Convert PyTorch tensors to NumPy for computation

    # Shapes
    n, m = x.shape[0], y.shape[0]
    indices = np.zeros((n, k), dtype=int)

    for i in tqdm(range(n)):
        curr_x = x[i].reshape(1, -1)  # Reshape to (1, feature_dim)

        # Compute pairwise distances
        if callable(metric):
            curr_dist = cdist(curr_x, y, metric=metric)[0]  # Use scipy's cdist for custom metric
        else:
            curr_dist = cdist(curr_x, y, metric=metric)[0]  # Default to predefined scipy metrics

        # Get indices of k smallest distances
        k_indices = np.argsort(curr_dist)[:k]
        indices[i] = k_indices

    if not is_numpy:
        indices = torch.tensor(indices, dtype=torch.int64)  # Convert back to PyTorch tensor

    return indices


def conv_numerical_to_ranks(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert numerical data to ranks.
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    if isinstance(x, list):
        x = np.array(x)
    ranks = np.empty_like(x)
    ranks[x.argsort()] = np.arange(len(x))
    return ranks


def conv_numerical_to_ranks_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Convert numerical data to ranks using PyTorch tensors.
    """
    sorted_indices = torch.argsort(x)
    ranks = torch.empty_like(sorted_indices, dtype=x.dtype)
    ranks[sorted_indices] = torch.arange(len(x), dtype=x.dtype, device=x.device)

    return ranks


def pairwise_distance_matrix(data1: torch.Tensor, data2: torch.Tensor = None,
                             metric: Union[str, Callable] = "euclidean", return_type: str = "distance",
                             verbose: bool = False):
    if return_type not in ["distance", "indices"]:
        raise NotImplementedError(f"return type {return_type} not supported")

    if data2 is None:
        data2 = data1

    nrow, ncol = data1.shape[0], data2.shape[0]
    metric_func = metrics_store(metric, "torch")
    matrics = torch.zeros((nrow, ncol), dtype=torch.int if return_type == "indices" else torch.float64)
    for i in tqdm(range(nrow), desc="calculating pairwise distances", disable=not verbose):
        curr_row = data1[i]
        curr_dist = metric_func(curr_row.expand((ncol, data1.shape[1])), data2)
        if return_type == "indices":
            matrics[i] = torch.argsort(curr_dist)
        else:
            matrics[i] = curr_dist

    return matrics


def row_wise_intersection_count(a, b):
    """
    Computes the number of intersections for each row in matrices a and b.

    Args:
        a (torch.Tensor): Tensor of shape (m, N).
        b (torch.Tensor): Tensor of shape (m, K).

    Returns:
        torch.Tensor: A tensor of shape (m,) where each element is the count of intersections for the corresponding row.
    """
    # Ensure inputs are 2D tensors
    if a.dim() != 2 or b.dim() != 2:
        raise ValueError("Both inputs must be 2D tensors.")

    # Expand dimensions to compare row-wise
    a_expanded = a.unsqueeze(2)  # Shape: (m, N, 1)
    b_expanded = b.unsqueeze(1)  # Shape: (m, 1, K)

    # Compare elements row-wise
    intersections = (a_expanded == b_expanded)  # Shape: (m, N, K)

    # Count intersections for each row
    intersection_count = intersections.sum(dim=(1, 2))  # Sum over N and K for each row

    return intersection_count


def metric_PN_at_K(original_data, embedding_data, N, K,
                   original_data_dist_mat = None, embedding_data_dist_mat = None,
                   metric: Union[str, Callable] = "euclidean"):
    assert original_data.shape[0] == embedding_data.shape[0]
    metric_func = metrics_store(metric, "torch")

    r, oc = original_data.shape
    _, ec = embedding_data.shape

    if original_data_dist_mat is None:
        original_data_dist_ind = pairwise_distance_matrix(original_data, metric=metric_func, return_type="indices")[:, :N]
    else:
        original_data_dist_ind = original_data_dist_mat

    if embedding_data_dist_mat is None:
        embedding_data_dist_ind = pairwise_distance_matrix(embedding_data, metric=metric_func, return_type="indices")[:, :K]
    else:
        embedding_data_dist_ind = embedding_data_dist_mat

    interceptions = row_wise_intersection_count(original_data_dist_ind, embedding_data_dist_ind) / N

    return interceptions.mean()

