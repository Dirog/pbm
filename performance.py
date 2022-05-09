import utils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
#######################################################################################
tx, rx, bits = utils.read_data('pbm.mat', 100)
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
M = np.arange(1, 5, 1)
BERs = []; SERs = []; NMSEs = []
for m in tqdm(M):
    fx = PBM(rx[:,0], rx[:,1], tx[:,0], m)
    fy = PBM(rx[:,1], rx[:,0], tx[:,1], m)

    rx_pbm = np.zeros(rx.shape, dtype=np.complex64)
    rx_pbm[:,0] = fx; rx_pbm[:,1] = fy
    pbm_ber, pbm_ser = utils.ber_and_ser(rx_pbm, bits)
    pbm_nmse = utils.nmse(rx_pbm, tx)

    BERs.append(pbm_ber)
    SERs.append(pbm_ser)
    NMSEs.append(pbm_nmse)

plt.figure()
plt.subplot(3,1,1)
plt.plot(M, BERs)
plt.xlabel('PBM max delay')
plt.ylabel('BER')
plt.grid()

plt.subplot(3,1,2)
plt.plot(M, SERs)
plt.xlabel('PBM max delay')
plt.ylabel('SER')
plt.grid()

plt.subplot(3,1,3)
plt.plot(M, NMSEs)
plt.xlabel('PBM max delay')
plt.ylabel('NMSE, dB')
plt.grid()

plt.tight_layout()
plt.show()
