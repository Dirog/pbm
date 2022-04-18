import qam
import utils
import scipy.io
import numpy as np
from math import log
import matplotlib.pyplot as plt

data = scipy.io.loadmat('pbm_test.mat')

M = 16
skip = 100
tx = data['srcSymData'][skip::, :]
rx = data['eqSymOutData'][skip::, :]
bits = data['srcPermBitData'][skip * 4::, :]

tx = tx[:tx.shape[0] - skip:, :]
rx = rx[:rx.shape[0] - skip:, :]
bits = bits[:bits.shape[0] - skip*4:,:]

means = utils.conditional_mean(rx, tx)
plt.scatter(np.real(rx[:,0]), np.imag(rx[:,0]), alpha=0.01)
plt.scatter(np.real(rx[:,1]), np.imag(rx[:,1]), alpha=0.01)
plt.scatter(np.real(means[:,0]), np.imag(means[:,0]), s=50, marker='x')
plt.scatter(np.real(means[:,1]), np.imag(means[:,1]), s=50, marker='x')
plt.legend(['RX X', 'RX Y', 'Mean X', 'Mean Y'], loc='upper right')
plt.grid()
plt.show()

constel = qam.constellation(M)
indexes = qam.demodulate(constel, means[:,0])
constel_x = means[indexes, 0]
constel_y = means[indexes, 1]
rx_data_x = qam.demodulate(rx[:, 0], constel)
rx_data_y = qam.demodulate(rx[:, 1], constel)


ber_x = 1 - np.mean(utils.bitarray(rx_data_x, 16) == bits[:, 0])
ber_y = 1 - np.mean(utils.bitarray(rx_data_y, 16) == bits[:, 1])

e = rx - tx
nmse_x = np.sum(np.abs(e[:,0])**2) / np.sum(np.abs(rx[:,0])**2)
nmse_y = np.sum(np.abs(e[:,1])**2) / np.sum(np.abs(rx[:,1])**2)

ser_x = 1 - np.mean(rx_data_x == utils.bits_to_ints(bits[:,0], M))
ser_y = 1 - np.mean(rx_data_y == utils.bits_to_ints(bits[:,1], M))

nmse = (nmse_x + nmse_y) / 2
ber = (ber_x + ber_y) / 2
ser = (ser_x + ser_y) / 2
metrics = {'BER' : ber, 'SER' : ser, 'NMSE (dB)': 10*np.log10(nmse)}
print(metrics)