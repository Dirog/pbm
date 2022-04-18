import numpy as np

def getbits(x, n):
    return list(map(lambda i: (i >> n) & 1, x))


def bitarray(x, max_int):
    bit_count = int(np.log2(max_int))
    result = np.zeros((len(x), bit_count), dtype=int)
    for i in range(bit_count):
        bits = getbits(x, bit_count - i - 1)
        result[:,i] = np.array(bits)

    return result.flatten()


def conditional_mean(rx, tx):
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