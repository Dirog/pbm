import utils
import numpy as np
import matplotlib.pyplot as plt
#######################################################################################
tx, rx, bits = utils.read_data('pbm.mat', 100)
init_means = utils.conditional_mean(rx, tx)
#######################################################################################
plot_initial = True
if plot_initial:
    plt.figure()
    plt.scatter(np.real(rx[:,0]), np.imag(rx[:,0]), alpha=0.02)
    plt.scatter(np.real(rx[:,1]), np.imag(rx[:,1]), alpha=0.02)
    plt.scatter(np.real(init_means[:,0]), np.imag(init_means[:,0]), marker="x")
    plt.scatter(np.real(init_means[:,1]), np.imag(init_means[:,1]), marker="x")
    plt.title('Initial X and Y pol.')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.grid()
    plt.show()
#######################################################################################
init_ber, init_ser = utils.ber_and_ser(rx, bits)
init_nmse = utils.nmse(rx, tx)
print('##################################')
print("Initial BER: {:e}".format(init_ber))
print("Initial SER: {:e}".format(init_ser))
print("Initial NMSE: {:.3f} dB".format(init_nmse))
#######################################################################################
def PBM(x, y, d, M):
    N = len(x); K = 2*M+1
    U = np.zeros((N, K, K, 2), dtype=np.complex64)

    xp = np.pad(x, (2*M, 2*M))
    yp = np.pad(y, (2*M, 2*M))
    for m in range(-M,M+1):
        for n in range(-M,M+1):
            i = n + M; j = m + M
            U[:,i,j,0] = xp[M+j:M+N+j] * np.conj(xp[j+i:N+j+i]) * xp[M+i:M+N+i]
            U[:,i,j,1] = xp[M+j:M+N+j] * np.conj(yp[j+i:N+j+i]) * yp[M+i:M+N+i] 
                       
    U = np.reshape(U, (N, -1))
    U = np.c_[x, U]

    c = np.linalg.lstsq(U, d, rcond=None)
    return U @ c[0]
#######################################################################################
M = 5
fx = PBM(rx[:,0], rx[:,1], tx[:,0], M)
fy = PBM(rx[:,1], rx[:,0], tx[:,1], M)
#######################################################################################
plot_after_pbm = True
if plot_after_pbm:
    plt.figure()
    plt.scatter(np.real(fx), np.imag(fx), alpha=0.02)
    plt.scatter(np.real(rx[:,0]), np.imag(rx[:,0]), alpha=0.02)
    plt.title('After PBM IQ (only X pol.)')
    plt.xlabel('I')
    plt.ylabel('Q')
    plt.grid()
    plt.show()
#######################################################################################
rx_pbm = np.zeros(rx.shape, dtype=np.complex64)
rx_pbm[:,0] = fx; rx_pbm[:,1] = fy
pbm_ber, pbm_ser = utils.ber_and_ser(rx_pbm, bits)
pbm_nmse = utils.nmse(rx_pbm, tx)
print('##################################')
print("Post PBM BER: {:e}".format(pbm_ber))
print("Post PBM SER: {:e}".format(pbm_ser))
print("Post PBM NMSE: {:.3f} dB".format(pbm_nmse))
print('##################################')
