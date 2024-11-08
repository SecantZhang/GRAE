# source: github.com/thedatumorg/kshape-python

import numpy as np
from numba import njit
from numba.types import float32 as nb_float32, Array
from typing import Union


@njit
def jit_fft(x, fft_size, axis=0): 
    return np.fft.fft(x, fft_size, axis=axis)


@njit
def jit_ifft(x, axis=0): 
    return np.fft.ifft(x, axis=axis)


jit_compile = True


@njit
def roll_zeropad(a, shift, axis=None):
    a = np.asanyarray(a)

    if shift == 0:
        return a

    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False

    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift, n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift, n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)

    if reshape:
        return res.reshape(a.shape)
    else:
        return res


@njit((Array(nb_float32, 2, 'C'), Array(nb_float32, 2, 'C')), nopython=True)
def ncc_c_3dim(x, y):
    den = np.sqrt((x ** 2).sum()) * np.sqrt((y ** 2).sum())

    if den < 1e-9:
        den = np.inf

    x_len = x.shape[0]
    
    # compute the next power of 2 in jit
    fft_size = 1
    while fft_size < 2 * x_len - 1:
        fft_size <<= 1

    cc = jit_ifft(jit_fft(x, fft_size, axis=0) * np.conj(jit_fft(y, fft_size, axis=0)), axis=0)
    cc = np.concatenate((cc[-(x_len-1):], cc[:x_len]), axis=0)

    return np.real(cc).sum(axis=-1) / den


@njit((Array(nb_float32, 2, 'C'), Array(nb_float32, 2, 'C')), nopython=True)
def sbd(x, y):
    ncc = ncc_c_3dim(x, y)
    value = ncc.max() # algorithm 1, dist 
    
    dist = 1 - value
    return dist


@njit((Array(nb_float32, 1, 'C'), Array(nb_float32, 1, 'C')), nopython=True)
def sbd_1d(x, y): 
    return sbd(x.reshape(-1, x.shape[0]), y.reshape(-1, y.shape[0]))


# testing
def print_list(li: Union[list, np.ndarray]) -> None:
    list_str = ", ".join(map(str, li))
    list_str = "[" + list_str + "]"
    print(list_str)


if __name__ == "__main__":
    np.random.seed(0)
    n, m = 3, 32
    test_size = 100
    score = []

    for i in range(test_size):
        data1 = np.random.rand(m).astype(np.float32)
        data2 = np.random.rand(m).astype(np.float32)
        score.append(sbd_1d(data1, data2))
        
    print(score)
    print(len(score))
