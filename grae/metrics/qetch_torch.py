import torch
import typing
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def calc_width(serie_1d: torch.Tensor) -> torch.Tensor:
    return torch.max(serie_1d) - torch.min(serie_1d)


class Segment:
    def __init__(self, serie):
        self.serie = serie
        self.width = torch.max(serie[:, 0]) - torch.min(serie[:, 0])
        self.ceil = torch.max(serie[:, 1])
        self.floor = torch.min(serie[:, 1])
        self.height = self.ceil - self.floor

    def __len__(self):
        return self.serie.shape[0]

    def getw(self):
        return self.width

    def geth(self):
        return self.height

    def get_boarder_left(self):
        return self.serie[0, 0]  # return leftmost x-value

    def get_boarder_right(self):
        return self.serie[-1, 0]  # return rightmost x-value

    def get_ceil(self):
        return self.ceil

    def get_floor(self):
        return self.floor

    def get_val_x(self, idx):
        return self.serie[idx, 0]

    def get_val_y(self, idx):
        return self.serie[idx, 1]  # get x-value of point at index idx


def combine_segments(left_seg: Segment, right_seg: Segment):
    new_series = torch.cat((left_seg.serie, right_seg.serie), dim=0)
    new_segment = Segment(new_series)
    return new_segment


def local_distortion_error(q_seg, c_seg, gx, gy):
    Rx = c_seg.getw() / (gx * q_seg.getw())
    Ry = c_seg.geth() / (gy * q_seg.geth())
    return torch.log(Rx) ** 2 + torch.log(Ry) ** 2


def shape_error(q_seg, c_seg, gy, hc):
    min_len = min(len(q_seg), len(c_seg))
    Ry = c_seg.height / (gy * q_seg.height)
    errors = torch.abs((gy * Ry * q_seg.serie[:min_len, 1] - c_seg.serie[:min_len, 1]) / hc)
    return errors.sum() / len(q_seg)


def local_distortion_errors(q_segs: typing.List[Segment], c_segs: typing.List[Segment], gx: torch.Tensor, gy: torch.Tensor) -> torch.Tensor:
    assert len(q_segs) == len(c_segs)
    errors = [local_distortion_error(q_segs[idx], c_segs[idx], gx, gy) for idx in range(len(q_segs))]
    return torch.sum(torch.stack(errors))


def shape_errors(q_segs: typing.List[Segment], c_segs: typing.List[Segment], gy: torch.Tensor, hc: torch.Tensor) -> torch.Tensor:
    assert len(q_segs) == len(c_segs)
    errors = [shape_error(q_segs[idx], c_segs[idx], gy, hc) for idx in range(len(q_segs))]
    return torch.sum(torch.stack(errors))


def exponential_moving_average(tensor: torch.Tensor, alpha: float = 0.9):
    ema = torch.zeros_like(tensor)
    ema[0] = tensor[0]
    for i in range(1, len(tensor)):
        ema[i] = alpha * ema[i - 1] + (1 - alpha) * tensor[i]

    return ema


def segmentation(series: torch.Tensor, int_num=128, ema_alpha=0.7, plot=False):
    # this code assumes the input series is already smoothed.
    if ema_alpha > 0:
        series = exponential_moving_average(series, alpha=ema_alpha)
    series_x = torch.arange(0, series.shape[0]).to(device)

    deriv = torch.gradient(series)[0]
    noise_threshold = 0.01 * torch.abs(torch.max(series) - torch.min(series))

    segment_indices = [0]
    for i in range(1, deriv.shape[0]):
        if torch.sign(deriv[i]) != torch.sign(deriv[i - 1]):
            segment_indices.append(i)
    segment_indices.append(deriv.shape[0])

    segments = []
    for i in range(len(segment_indices) - 1):
        start, end = segment_indices[i], segment_indices[i + 1]
        segment = series[start:end]
        if torch.abs(torch.max(segment) - torch.min(segment)) > noise_threshold:
            segments.append(Segment(torch.stack([series_x[start:end], segment], dim=1)))

    if plot:
        segmentation_vals = [segment.serie[0, 0] for segment in segments] + [segments[len(segments)-1].serie[-1, 0]]
        plt.plot(series, color="blue", alpha=0.4)
        plt.plot(series_x, series, color="green", alpha=0.4)
        plt.vlines(segmentation_vals, color="red", alpha=0.4, linestyles="dashed", ymin=torch.min(series), ymax=torch.max(series))
        plt.show()

    return segments


def qetch_calculate(q_segs, c_segs):
    assert len(q_segs) == len(c_segs)
    q_width = q_segs[-1].serie[-1, 0] - q_segs[0].serie[0, 0]
    q_height = torch.max(torch.stack([segment.ceil for segment in q_segs])) - torch.min(torch.stack([segment.floor for segment in q_segs]))

    c_width = c_segs[-1].serie[-1, 0] - c_segs[0].serie[0, 0]
    c_height = torch.max(torch.stack([segment.ceil for segment in c_segs])) - torch.min(torch.stack([segment.floor for segment in c_segs]))

    Gx = c_width / q_width
    Gy = c_height / q_height

    ldr = local_distortion_errors(q_segs, c_segs, gx=Gx, gy=Gy)
    se = shape_errors(q_segs, c_segs, gy=Gy, hc=c_height)

    return ldr + se

def qetch_search(s1, s2):
    s1_segments = segmentation(s1)
    s2_segments = segmentation(s2)
    if len(s1_segments) < len(s2_segments):
        shorter_segments, longer_segments = s1_segments, s2_segments
    else:
        shorter_segments, longer_segments = s2_segments, s1_segments
    offset = len(longer_segments) - len(shorter_segments) + 1

    distances = torch.tensor([qetch_calculate(shorter_segments, longer_segments[i:i + len(shorter_segments)]) for i in range(offset)])

    return distances.min()


def qetch_search_batched(s1_batch, s2_batch):
    assert len(s1_batch) == len(s2_batch)
    num_batches = len(s1_batch)
    results = []

    for batch_idx in range(num_batches):
        distance = qetch_search(s1_batch[batch_idx], s2_batch[batch_idx])
        distance.requires_grad_()
        results.append(distance)

    results = torch.stack(results)
    results.requires_grad_()
    return results
