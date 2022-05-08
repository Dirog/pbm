import qam
import scipy.io
import numpy as np


def read_data(path : str, skip : int):
    data = scipy.io.loadmat(path)
    tx = data['srcSymData'][skip::, :]
    rx = data['eqSymOutData'][skip::, :]
    bits = data['srcPermBitData'][skip * 4::, :]
    bits = bits[:bits.shape[0] - skip*4:,:]

    tx = tx[:tx.shape[0] - skip:, :]
    rx = rx[:rx.shape[0] - skip:, :]

    return tx, rx, bits


def conditional_mean(rx : np.array(np.complex64),
                     tx : np.array(np.complex64)) -> np.array(np.complex64):

    constel = np.unique(tx)
    count = len(constel)
    means = np.zeros((count, 2), dtype=complex)
    for i in range(count):
        for p in range(2):
            rx_p = rx[:,p]
            tx_p = tx[:,p]
            point = constel[i]
            mean = np.mean(rx_p[tx_p == point])
            means[i, p] = mean

    return means


def getbits(x : np.array(int), n : int) -> np.array(int):
    return list(map(lambda i: (i >> n) & 1, x))


def bitarray(x : np.array(int), max_int : int) -> np.array(int):
    bit_count = int(np.log2(max_int))
    result = np.zeros((len(x), bit_count), dtype=int)
    for i in range(bit_count):
        bits = getbits(x, bit_count - i - 1)
        result[:,i] = np.array(bits)

    return result.flatten()


def bits_to_ints(bit_array, max_int):
    bit_count = len(bit_array)
    bit_per_int = int(np.log2(max_int))
    int_count = bit_count // bit_per_int

    if int_count * bit_per_int != bit_count:
        raise ValueError("Bad bit array size!")

    ints = np.zeros((int_count,), dtype=int)
    for i in range(int_count):
        bits = bit_array[i * bit_per_int:(i+1)*bit_per_int]
        out = 0
        for bit in bits:
            out = (out << 1) | bit
        ints[i] = out

    return ints


def ber_and_ser(rx : np.array(np.complex64), bits : np.array(int)) -> float:

    M = 16
    constel = qam.constellation(M)
    #means = conditional_mean(rx, tx)
    
    #indexes = qam.demodulate(constel, means[:,0])
    #constel_x = means[indexes, 0]
    #constel_y = means[indexes, 1]
    rx_data_x = qam.demodulate(rx[:, 0], constel)
    rx_data_y = qam.demodulate(rx[:, 1], constel)

    ber_x = 1 - np.mean(bitarray(rx_data_x, 16) == bits[:, 0])
    ber_y = 1 - np.mean(bitarray(rx_data_y, 16) == bits[:, 1])
    ser_x = 1 - np.mean(rx_data_x == bits_to_ints(bits[:,0], M))
    ser_y = 1 - np.mean(rx_data_y == bits_to_ints(bits[:,1], M))

    return (ber_x + ber_y) / 2, (ser_x + ser_y) / 2


def nmse(rx : np.array(np.complex64), tx : np.array(np.complex64)) -> float:
    e = rx - tx

    nmse_x = np.sum(np.abs(e[:,0])**2) / np.sum(np.abs(rx[:,0])**2)
    nmse_y = np.sum(np.abs(e[:,1])**2) / np.sum(np.abs(rx[:,1])**2)

    return 10*np.log10(( nmse_x + nmse_y ) / 2)



