import numpy as np
from math import log


def modulate(x : np.array(int), constel : np.array(np.complex64)) -> np.array(np.complex64):
    return constel[x]


def demodulate(x : np.array(np.complex64), constel : np.array(np.complex64)) -> np.array(int):
    samples = len(x)
    result = np.zeros(x.shape)
    for i in range(samples):
        j = np.argmin(np.abs(constel - x[i]))
        result[i] = j
    
    return result.astype(int)


def constellation(M : int) -> np.array(np.complex64):
    if np.fix(log(M, 4)) != log(M,4):
        raise ValueError("M must be power of 4!")

    nbits = int(np.log2(M))
    x = np.arange(M)

    nbitsBy2 = nbits >> 1
    symbolI = x >> nbitsBy2
    symbolQ = x & ((M-1) >> nbitsBy2)

    i = 1
    while i < nbitsBy2:
        tmpI = symbolI
        tmpI = tmpI >> i
        symbolI = symbolI ^ tmpI

        tmpQ = symbolQ
        tmpQ = tmpQ >> i
        symbolQ = symbolQ ^ tmpQ
        i = i + i

    gray = (symbolI << nbitsBy2) + symbolQ

    x = x[gray]
    c = int(np.sqrt(M))
    I = -2 * np.mod(x, c) + c - 1
    Q = 2 * np.floor(x / c) - c + 1
    IQ = I + 1j*Q
    IQ = -np.transpose(np.reshape(IQ, (c, c)))
    return IQ.flatten()