import numpy as np
import matplotlib.pyplot as plt
from numba import jit
from math import log
from typing import Union


jit_compile = False


def jit_compilation(func):
    """
    helper function for debugging, switch on or off for jit compilation
    """
    if jit_compile:
        return jit(func, nopython=True)
    return func


def plot(data, derivative_points, prefix="test"):
    num_rows = data.shape[0]
    fig, axs = plt.subplots(num_rows, 1, figsize=(10, 2 * num_rows), sharex=True)
    if num_rows == 1:
        axs = [axs]

    for i in range(num_rows):
        axs[i].plot(data[i], label='Data')
        axs[i].set_title(f'Row {i + 1}')
        axs[i].set_ylabel('Value')
        for vline in derivative_points[i]:
            axs[i].axvline(x=vline, color='r', linestyle='--', label='Derivative Point')

    axs[-1].set_xlabel('Index')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{prefix}.png")
    plt.show()


@jit_compilation
def compute_diff_2d(s: np.ndarray):
    """
    Custom implementation of np.diff with axis=1
    """
    n, m = s.shape
    diff = np.empty((n, m - 1), dtype=s.dtype)
    for i in range(n):
        for j in range(m - 1):
            diff[i, j] = s[i, j + 1] - s[i, j]
    return diff


@jit_compilation
def compute_diff_1d(s: np.ndarray) -> np.ndarray:
    """
    Custom implementation of np.diff for a 1-dimensional vector.
    :param s: input ndarray, shape (d)
    :return: difference array of shape (d-1)
    """
    m = s.shape[0]
    diff = np.empty(m - 1, dtype=s.dtype)
    for j in range(m - 1):
        diff[j] = s[j + 1] - s[j]
    return diff


@jit_compilation
def linear_interpolate(x, xp, fp):
    """
    Perform linear interpolation for each point in x based on xp and fp.
    :param x: The x-coordinates at which to interpolate
    :param xp: The x-coordinates of the data points
    :param fp: The y-coordinates of the data points
    :return: The interpolated values
    """
    interpolated_values = np.zeros_like(x, dtype=np.float32)
    for i in range(len(x)):
        for j in range(len(xp) - 1):
            if xp[j] <= x[i] <= xp[j + 1]:
                interpolated_values[i] = fp[j] + (fp[j + 1] - fp[j]) * (x[i] - xp[j]) / (xp[j + 1] - xp[j])
                break
    return interpolated_values


@jit_compilation
def interpolate_vector(vec, new_length):
    """
    Interpolate a vector to a new length using linear interpolation.
    :param vec: The original vector
    :param new_length: The desired length of the interpolated vector
    :return: Interpolated vector of length new_length
    """
    original_indices = np.linspace(0, 1, len(vec))
    new_indices = np.linspace(0, 1, new_length)
    return linear_interpolate(new_indices, original_indices, vec)


@jit_compilation
def manhattan_distance_interpolated(vec1, vec2):
    """
    Calculate the Manhattan distance between two vectors with unequal lengths
    by interpolating the shorter vector.
    :param vec1: The first vector
    :param vec2: The second vector
    :return: The Manhattan distance between the two vectors
    """
    len1, len2 = len(vec1), len(vec2)

    if len1 < len2:
        vec1 = interpolate_vector(vec1, len2)
    elif len2 < len1:
        vec2 = interpolate_vector(vec2, len1)

    distance = np.sum(np.abs(vec1 - vec2))
    return distance


@jit_compilation
def segmentations(s: np.ndarray, noise_threshold: float = 0.01):
    """
    Segmentation by derivative sign
    :param noise_threshold: Threshold to ignore small variations
    :param s: input ndarray in shape (d)
    :return: all_segments, segment_points, segments_wh, global_wh
    """
    global_wh = (s.shape[0], np.max(s) - np.min(s))
    threshold = global_wh[1] * noise_threshold
    # print(global_wh)
    derivatives = compute_diff_1d(s)
    sign_changes = np.where(np.diff(np.sign(derivatives)))[0] + 1
    curr_segments = []
    curr_segments_wh = []
    start_idx = 0
    for idx in sign_changes:
        if idx > start_idx:
            curr_segment = s[start_idx: idx+1]
            curr_segment_height = max(curr_segment) - min(curr_segment)
            if curr_segment_height <= threshold:
                # print(f"sign_change: {idx}")
                # curr_segments[-1] = curr_segment[-1] + curr_segment
                # curr_segments_wh[-1] = (len(curr_segments[-1]), max(curr_segments[-1]) - min(curr_segments[-1]))
                continue
            else:
                curr_segments.append(curr_segment)
                curr_segments_wh.append((len(curr_segment), curr_segment_height))
            start_idx = idx + 1
    if start_idx < len(s):
        curr_segments.append(s[start_idx:])
        curr_segments_wh.append((len(curr_segments[-1]), max(curr_segments[-1]) - min(curr_segments[-1])))

    return curr_segments, curr_segments_wh, global_wh


@jit_compilation
def local_distortion_error(q_segs_wh: list, c_segs_wh: list, gx: float, gy: float):
    """
    compute the local distortion errors in page 6
    assumption is that the q_segs_wh has equal length as c_segs_wh
    :param q_segs_wh: list of width and height for query segment
    :param c_segs_wh: list of width and height for candidate segment
    :param gx: Gx = width(C) / width(Q)
    :param gy: Gy = height(C) / height(Q)
    :return:
    """
    rx = sum([log(c_segs_wh[i][0] / (gx * q_segs_wh[i][0])) ** 2 for i in range(len(q_segs_wh))])
    ry = sum([log(c_segs_wh[i][1] / (gy * q_segs_wh[i][1])) ** 2 for i in range(len(q_segs_wh)) if c_segs_wh[i][1] != 0 and (gy * q_segs_wh[i][1]) ** 2 != 0])

    return rx + ry


@jit_compilation
def shape_error(q_segs: list, c_segs: list, q_segs_wh: list, c_segs_wh: list, gy: float, ch: list):
    se = 0
    for segs_idx in range(len(q_segs)):
        if len(q_segs[segs_idx]) < len(c_segs[segs_idx]):
            q_segs[segs_idx] = interpolate_vector(q_segs[segs_idx], len(c_segs[segs_idx]))
        elif len(q_segs[segs_idx]) > len(c_segs[segs_idx]):
            c_segs[segs_idx] = interpolate_vector(c_segs[segs_idx], len(c_segs[segs_idx]))

        if ch[segs_idx] != 0:
            se += (gy * (c_segs_wh[segs_idx][1] / gy * q_segs_wh[segs_idx][1]) * q_segs_wh[segs_idx][1] - c_segs_wh[segs_idx][1]) / ch[segs_idx]
    return se / len(q_segs)


@jit_compilation
def qetch(s1: np.ndarray, s2: np.ndarray):
    segs_s1, segs_wh_s1, global_wh_s1 = segmentations(s1)
    segs_s2, segs_wh_s2, global_wh_s2 = segmentations(s2)

    if len(segs_s1) <= len(segs_s2):
        segs_s, segs_wh_s = segs_s1, segs_wh_s1
        segs_l, segs_wh_l = segs_s2, segs_wh_s2
        global_wh_s = global_wh_s1
        global_wh_l = global_wh_s2
    else:
        segs_s, segs_wh_s = segs_s2, segs_wh_s2
        segs_l, segs_wh_l = segs_s1, segs_wh_s1
        global_wh_s = global_wh_s2
        global_wh_l = global_wh_s1

    offset = abs(len(segs_s1) - len(segs_s2)) + 1
    # print(offset)

    gx = global_wh_l[0] / global_wh_s[0]
    gy = global_wh_l[1] / global_wh_s[1]

    qetch_scores = [-1 for _ in range(offset)]
    for i in range(offset):
        sliding_window_segs = segs_l[i:i + len(segs_s)]
        sliding_window_wh   = segs_wh_l[i:i + len(segs_s)]
        # ch = max([wh[1] for wh in sliding_window_wh]) - min([wh[1] for wh in sliding_window_wh])
        ch = [wh[1] for wh in sliding_window_wh]
        se = shape_error(segs_s, sliding_window_segs, segs_wh_s, sliding_window_wh, gy, ch)
        lde = local_distortion_error(segs_wh_s, sliding_window_wh, gx, gy)
        qetch_scores[i] = lde + se

    # print(qetch_scores)
    return min(qetch_scores)


def print_list(li: Union[list, np.ndarray]) -> None:
    list_str = ", ".join(map(str, li))
    list_str = "[" + list_str + "]"
    print(list_str)


if __name__ == "__main__":
    n, m = 3, 32
    test_size = 100
    qetch_score = []
    for i in range(test_size):
        data1 = np.random.rand(m).astype(np.float32)
        data2 = np.random.rand(m).astype(np.float32)
        qetch_score.append(qetch(data1, data2))

    print_list(qetch_score)
