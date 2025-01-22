import torch

def fft_tensor(x, fft_size, dim=0):
    """
    Batched FFT using PyTorch.
    """
    return torch.fft.fft(x, n=fft_size, dim=dim)

def ifft_tensor(x, dim=0):
    """
    Batched IFFT using PyTorch.
    """
    return torch.fft.ifft(x, dim=dim)

def roll_zeropad(tensor, shift, dim=-1):
    """
    Roll with zero-padding in PyTorch.
    """
    n = tensor.size(dim)
    if shift == 0 or abs(shift) > n:
        return torch.zeros_like(tensor)
    if shift < 0:
        shift += n
    zeros = torch.zeros_like(tensor.narrow(dim, 0, n - shift))
    rolled = torch.cat((tensor.narrow(dim, n - shift, shift), zeros), dim=dim)
    return rolled

def ncc_c_3dim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute normalized cross-correlation for batched input.
    :param x: Tensor of shape (batch_size, sequence_length)
    :param y: Tensor of shape (batch_size, sequence_length)
    :return: Tensor of shape (batch_size, 2*sequence_length - 1)
    """
    if len(x.size()) == 3:
        # TODO: Need fix this special case
        _, batch_size, seq_len = x.size()
    else:
        batch_size, seq_len = x.size()
    den = torch.sqrt((x**2).sum(dim=1) * (y**2).sum(dim=1)).unsqueeze(1)

    # Avoid division by zero
    den[den < 1e-9] = float('inf')

    fft_size = 1
    while fft_size < 2 * seq_len - 1:
        fft_size <<= 1

    # Perform FFT and IFFT
    cc = ifft_tensor(
        fft_tensor(x, fft_size, dim=1) * torch.conj(fft_tensor(y, fft_size, dim=1)),
        dim=1
    )

    # Concatenate and normalize
    cc = torch.cat((cc[:, -(seq_len - 1):], cc[:, :seq_len]), dim=1)
    cc_real = cc.real.sum(dim=-1) / den
    return cc_real

def sbd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the SBD distance for batched input.
    :param x: Tensor of shape (batch_size, sequence_length)
    :param y: Tensor of shape (batch_size, sequence_length)
    :return: Tensor of shape (batch_size,)
    """
    ncc = ncc_c_3dim(x, y)
    value, _ = ncc.max(dim=1)  # Algorithm 1: dist
    dist = 1 - value
    return dist

def sbd_1d(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    SBD for single-dimensional inputs, reshaped for batched computation.
    :param x: Tensor of shape (sequence_length,)
    :param y: Tensor of shape (sequence_length,)
    :return: Single scalar distance.
    """
    return sbd(x.unsqueeze(0), y.unsqueeze(0)).item()

# Testing
if __name__ == "__main__":
    torch.manual_seed(0)
    batch_s1 = torch.rand(10, 32)  # 10 sequences of length 32
    batch_s2 = torch.rand(10, 32)

    distances = sbd(batch_s1, batch_s2)  # Output: Tensor of shape (10,)
    print(distances)