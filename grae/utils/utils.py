import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Optional
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean


# def metric_function(metric_name: str) -> Callable:
#     if metric_name in ["l2", "euclidean"]:
#         return euclidean_metric
#     elif metric_name in ["dtw", "DTW"]:
#         return dtw_metric
#     else:
#         raise NotImplementedError()


# def euclidean_metric(x, y):
#     return np.linalg.norm(x-y)


def pairwise_dist_indices_calculation(x: np.ndarray,
                                      y: Optional[np.ndarray] = None,
                                      k: int = 50,
                                      metric: Callable = euclidean
                                      ) -> np.ndarray:
    y = x if y is None else y
    n, m = x.shape[0], y.shape[0]
    indices = np.zeros((n, k), dtype=int)

    for i in tqdm(range(n)):
        curr_x = x[i]
        curr_dist = pairwise_distances([curr_x], y, metric=metric, n_jobs=-1)[0]
        k_indices = np.argsort(curr_dist)[:k]
        indices[i] = k_indices

    return indices


def row_score(row_x: np.ndarray, row_y: np.ndarray) -> int:
    common_sets = np.intersect1d(row_x, row_y)
    return len(common_sets)


def indices_matrix_scores(mat_x: np.ndarray, mat_y: np.ndarray):
    assert mat_x.shape[0] == mat_y.shape[0]

    scores = np.zeros(mat_x.shape[0], dtype=int)
    for i in range(mat_x.shape[0]):
        scores[i] = row_score(mat_x[i], mat_y[i])

    return scores


def ranged_indices_matrix_scores(query_mat: np.ndarray, gt_mat: np.ndarray, limit: int, gt_k: int):
    assert query_mat.shape[0] == gt_mat.shape[0]

    score_mat = np.zeros((query_mat.shape[0], limit))
    for curr_slice in tqdm(range(1, limit + 1), desc="calculating scores"):
        sliced_query_mat = query_mat[:, :curr_slice]
        curr_scores = indices_matrix_scores(sliced_query_mat, gt_mat)
        score_mat[:, curr_slice-1] = curr_scores

    p_at_k_score = np.mean(indices_matrix_scores(gt_mat[:, 0], query_mat)) / gt_k
    rp_at_k_score = np.mean(indices_matrix_scores(gt_mat, query_mat)) / gt_k
    
    return score_mat, p_at_k_score, rp_at_k_score


def five_number_summary(data):
    if len(data) == 0:
        raise ValueError("Data list is empty")

    minimum = np.min(data)
    q1 = np.percentile(data, 25)
    median = np.median(data)
    q3 = np.percentile(data, 75)
    maximum = np.max(data)
    summary = minimum, q1, median, q3, maximum

    return [round(x, 3) for x in summary]


def scores_heatmap(scores_mat: np.ndarray, denom: int , p_at_k_score: float = None, rp_at_k_score: float = None, 
                   prefix: str = "", save: bool = True, show: bool = False, save_path: str = None):
    avg_scores = np.mean(scores_mat, axis=0) / denom

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [4, 1]})

    sns.heatmap(scores_mat, annot=False, cmap="YlGnBu", cbar=True, xticklabels=list(range(1, scores_mat.shape[1])), ax=ax1)
    ax1.set_title(f"Overlap heatmap: {prefix}, overall coverage: {round(np.mean(avg_scores), 2)}\n \
                        summary: {five_number_summary(avg_scores)}")
    ax1.set_xlabel("k")

    ax2.plot(list(range(1, scores_mat.shape[1] + 1)), avg_scores, marker='o', linestyle='-', color='b', markersize=4)
    if p_at_k_score is not None: 
        ax2.axhline(y=p_at_k_score, color="r", linestyle="--")
    if rp_at_k_score is not None: 
        ax2.axhline(y=rp_at_k_score, color="g", linestyle="--")
    ax2.set_title('Scores over k')
    ax2.set_xlabel('k')
    ax2.set_ylabel('Score')
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    if show:
        plt.show()
    if save:
        if save_path is None: 
            plt.savefig(f"benchmarks/fig_{prefix}.png")
        else: 
            plt.savefig(f"{save_path}/{prefix}.png")
