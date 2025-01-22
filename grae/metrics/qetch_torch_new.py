import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from math import log
from typing import Union

def plot(data, derivative_points, prefix="test"):
    num_rows = data.shape[0]
    fig, axs = plt.subplots(num_rows, 1, figsize=(10, 2 * num_rows), sharex=True)
    if num_rows == 1:
        axs = [axs]

    for i in range(num_rows):
        axs[i].plot(data[i].cpu().numpy(), label='Data')
        axs[i].set_title(f'Row {i + 1}')
        axs[i].set_ylabel('Value')
        for vline in derivative_points[i]:
            axs[i].axvline(x=vline, color='r', linestyle='--', label='Derivative Point')

    axs[-1].set_xlabel('Index')
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{prefix}.png")
    plt.show()

def compute_diff_2d(s: torch.Tensor) -> torch.Tensor:
    """
    Custom implementation of torch.diff with axis=1
    """
    return s[:, 1:] - s[:, :-1]

def compute_diff_1d(s: torch.Tensor) -> torch.Tensor:
    """
    Custom implementation of torch.diff for a 1-dimensional vector.
    """
    return s[1:] - s[:-1]

def linear_interpolate(x, xp, fp):
    """
    Perform linear interpolation for each point in x based on xp and fp without using `torch.interp`.
    """
    # Expand dimensions for broadcasting
    xp = xp.unsqueeze(0)
    fp = fp.unsqueeze(0)
    x = x.unsqueeze(1)

    # Calculate weights for interpolation
    left = torch.searchsorted(xp[0], x, right=True) - 1
    left = torch.clamp(left, 0, len(xp[0]) - 2)  # Ensure valid indices
    right = left + 1

    x0, x1 = xp[0][left], xp[0][right]
    y0, y1 = fp[0][left], fp[0][right]

    # Linear interpolation formula
    result = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return result.squeeze(1)

def interpolate_vector(vec, new_length):
    """
    Interpolate a vector to a new length using PyTorch's interpolate function.
    """
    # Reshape the vector to match the input shape expected by interpolate
    vec = vec.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    # Perform interpolation
    interpolated = F.interpolate(vec, size=new_length, mode='linear', align_corners=True)
    return interpolated.squeeze(0).squeeze(0)  # Remove batch and channel dimensions

def manhattan_distance_interpolated(vec1, vec2):
    """
    Calculate the Manhattan distance between two vectors with unequal lengths
    by interpolating the shorter vector.
    """
    len1, len2 = len(vec1), len(vec2)

    if len1 < len2:
        vec1 = interpolate_vector(vec1, len2)
    elif len2 < len1:
        vec2 = interpolate_vector(vec2, len1)

    return torch.sum(torch.abs(vec1 - vec2))

def segmentations(s: torch.Tensor, noise_threshold: float = 0.01):
    """
    Segmentation by derivative sign
    """
    global_wh = (s.shape[0], torch.max(s) - torch.min(s))
    threshold = global_wh[1] * noise_threshold
    derivatives = compute_diff_1d(s)
    sign_changes = torch.where(torch.diff(torch.sign(derivatives)))[0] + 1
    curr_segments = []
    curr_segments_wh = []
    start_idx = 0
    for idx in sign_changes:
        if idx > start_idx:
            curr_segment = s[start_idx: idx+1]
            curr_segment_height = torch.max(curr_segment) - torch.min(curr_segment)
            if curr_segment_height <= threshold:
                continue
            else:
                curr_segments.append(curr_segment)
                curr_segments_wh.append((len(curr_segment), curr_segment_height))
            start_idx = idx + 1
    if start_idx < len(s):
        curr_segments.append(s[start_idx:])
        curr_segments_wh.append((len(curr_segments[-1]), torch.max(curr_segments[-1]) - torch.min(curr_segments[-1])))

    return curr_segments, curr_segments_wh, global_wh

def local_distortion_error(q_segs_wh, c_segs_wh, gx, gy):
    """
    Compute the local distortion errors
    """
    rx = sum([log(c_segs_wh[i][0] / (gx * q_segs_wh[i][0])) ** 2 for i in range(len(q_segs_wh))])
    ry = sum([log(c_segs_wh[i][1] / (gy * q_segs_wh[i][1])) ** 2
              for i in range(len(q_segs_wh))
              if c_segs_wh[i][1] != 0 and (gy * q_segs_wh[i][1]) ** 2 != 0])

    return rx + ry

def shape_error(q_segs, c_segs, q_segs_wh, c_segs_wh, gy, ch):
    """
    Compute shape error
    """
    se = 0
    for segs_idx in range(len(q_segs)):
        if len(q_segs[segs_idx]) < len(c_segs[segs_idx]):
            q_segs[segs_idx] = interpolate_vector(q_segs[segs_idx], len(c_segs[segs_idx]))
        elif len(q_segs[segs_idx]) > len(c_segs[segs_idx]):
            c_segs[segs_idx] = interpolate_vector(c_segs[segs_idx], len(q_segs[segs_idx]))

        if ch[segs_idx] != 0:
            se += (gy * (c_segs_wh[segs_idx][1] / gy * q_segs_wh[segs_idx][1]) * q_segs_wh[segs_idx][1] - c_segs_wh[segs_idx][1]) / ch[segs_idx]
    return se / len(q_segs)

def qetch(s1: torch.Tensor, s2: torch.Tensor):
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
    gx = global_wh_l[0] / global_wh_s[0]
    gy = global_wh_l[1] / global_wh_s[1]

    qetch_scores = [-1 for _ in range(offset)]
    for i in range(offset):
        sliding_window_segs = segs_l[i:i + len(segs_s)]
        sliding_window_wh   = segs_wh_l[i:i + len(segs_s)]
        ch = [wh[1] for wh in sliding_window_wh]
        se = shape_error(segs_s, sliding_window_segs, segs_wh_s, sliding_window_wh, gy, ch)
        lde = local_distortion_error(segs_wh_s, sliding_window_wh, gx, gy)
        qetch_scores[i] = lde + se

    return abs(min(qetch_scores))


def qetch_batched(batch_s1: torch.Tensor, batch_s2: torch.Tensor) -> torch.Tensor:
    batch_size = batch_s1.shape[0]
    # Batch computations for all pairs
    return torch.tensor([qetch(batch_s1[b], batch_s2[b]) for b in range(batch_size)], device=batch_s1.device)


def print_list(li: Union[list, torch.Tensor]) -> None:
    list_str = ", ".join(map(str, li))
    list_str = "[" + list_str + "]"
    print(list_str)

if __name__ == "__main__":
    n, m = 3, 32
    test_size = 100
    qetch_score = []
    for i in range(test_size):
        data1 = torch.rand(m, dtype=torch.float32)
        data2 = torch.rand(m, dtype=torch.float32)
        qetch_score.append(qetch(data1, data2))

    print_list(qetch_score)