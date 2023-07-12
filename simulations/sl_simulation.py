import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt
import scipy.io as sio




def convert_back_to_time(fyo, Nsamples, freq_indtest):
    NumUniquePts = np.ceil((Nsamples + 1) / 2)

    if freq_indtest == -1:
        freq_indtest = np.arange(1, NumUniquePts + 1)

    full_fyo = np.zeros(Nsamples)
    full_fyo[freq_indtest - 1] = fyo

    tmp = np.zeros_like(full_fyo)
    tmp[:len(full_fyo)] = full_fyo
    tmp[len(NumUniquePts):] = tmp[len(NumUniquePts):] + np.flip(full_fyo[1:])

    y = np.real(np.fft.ifft(tmp, 'symmetric'))
    
    return y

def bandpass(y, freqrange, fres, do_plot=False):
    if not do_plot:
        do_plot = False

    Nsamples = len(y)
    Nunique_points = np.ceil((Nsamples + 1) / 2)
    fHz = np.arange(Nunique_points) * fres / Nsamples

    freq_ind = np.where((fHz >= freqrange[0]) & (fHz <= freqrange[1]))[0]

    fy = np.fft.fft(y)
    fyo = fy[freq_ind]

    x = convert_back_to_time(fyo, Nsamples, freq_ind)

    if do_plot:
        fy2 = np.zeros_like(fy)
        fy2[freq_ind] = fyo
        plt.figure()
        plt.plot(fHz[:len(fy2)], np.abs(fy2[:len(fHz)]))
        plt.figure()
        plt.plot(x)
        plt.hold(True)
        plt.plot(y, 'r--')
    
    return x


# Load the Structural Connectivity
data = sio.loadmat('SC_90AAL_32HCP.mat')

# Access the variables in the .mat file
C = data['C']  # Access the variable 'C' from the .mat file
D = data['D']  # Access the variable 'D' from the .mat file

# Convert the variables to numpy arrays
C = np.array(C)
D = np.array(D)
N = C.shape[0]
C = C / np.mean(C[np.ones(N) - np.eye(N) > 0])
D = D / 1000  # Convert Distance matrix to meters

# Define Model Parameters
K = 10  # Global Coupling Weight
MD = 0.005  # Mean delay scaling the distance matrix D
f = 40  # Fundamental frequency of the oscillators
a = -5  # Position of the units with respect to the bifurcation
sig_noise = 0.001  # Standard deviation of white noise added to each unit

# Define Simulation Parameters
tmax = 25  # Total simulated time in seconds
t_prev = 0.5  # Transient in seconds
dt = 1e-4  # Resolution of model integration in seconds
dt_save = 2e-3  # Resolution of saved brain activity in seconds

iomega = 1j * 2 * np.pi * f * np.ones(N)  # Complex frequency in radians/second
kC = K * C * dt  # Scale matrix C with 'K' and 'dt'
dsig = np.sqrt(dt) * sig_noise  # Normalize std of noise with dt

if MD == 0:
    Delays = np.zeros((N, N))
else:
    Delays = np.round((D / np.mean(D[C > 0]) * MD) / dt)
Delays[C == 0] = 0
Max_History = np.max(Delays) + 1
Delays = Max_History - Delays

Z = dt * np.random.randn(N, int(Max_History)) + 1j * dt * np.random.randn(N, int(Max_History))
Zsave = np.zeros((N, int(tmax / dt_save)))
sumz = np.zeros(N).reshape(-1,1)

nt = 0
for t in np.arange(dt, t_prev + tmax, dt):
    Znow = Z[:, -1].reshape(-1,1)

    dz = Znow * (a + iomega - np.abs(Znow ** 2)) * dt

    for n in range(N):
        sumz[n] = np.sum(kC[n, :] * (Z[((np.arange(1, N+1)) + (N * (Delays[n, :] - 1))).astype(int)] - Znow[n]))

    if MD:
        Z[:, :-1] = Z[:, 1:]

    Z[:, -1] = Znow + dz + sumz + dsig * np.random.randn(N) + 1j * dsig * np.random.randn(N)

    if np.mod(t, dt_save) == 0 and t > t_prev:
        nt += 1
        Zsave[:, nt - 1] = Z[:, -1]

# Load the .mat file
mat = sio.loadmat('AAL_labels.mat')

# Access the variable containing the character array
label90 = mat['label90']

# Convert the character array to a numpy array of strings
label90 = label90.astype(str)
label90 = label90.reshape((90, 3))

plt.figure()
plt.plot(np.arange(0, (len(Zsave) - 1) * dt_save, dt_save), (np.arange(1, N + 1) * np.ones_like(Zsave)) + (np.real(Zsave) / (4 * np.std(Zsave, axis=1)[:, None])), 'k')
plt.title(f'Raw simulated signals in {N} units coupled with K={K} and MD={MD * 1000}ms', fontsize=14)
plt.xlabel('Time (seconds)', fontsize=18)
plt.box(False)
plt.axis('tight')
plt.gca().set_yticks(np.arange(1, N + 1))
plt.gca().set_yticklabels([])
plt.gca().set_yticklabels(label90[::-1, :], fontsize=6)
plt.ylim(0, N + 1)
plt.xlim(0, tmax)

# Plot the Power spectrum of all the nodes
Fourier_Power = np.zeros_like(Zsave)
freq_axis = np.arange(len(Zsave)) / (dt_save * len(Zsave))
for n in range(N):
    Fourier_Power[n, :] = np.abs(np.fft.fft(Zsave[n, :]))

# Note that here Zsave is complex so the Power Spectrum is not mirrored
plt.figure()
plt.plot(freq_axis, np.mean(Fourier_Power, axis=0))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.xlim(0, 100)

# Bandpass at a given range to see if collective oscillations are detected
high_pass = 0.5  # Define the slowest frequency to pass (in Hertz)
low_pass = 25  # Define the fastest frequency to pass (in Hertz)

Zfilt = np.zeros_like(Zsave)
for n in range(N):
    b, a = butter(2, [high_pass, low_pass], btype='band', fs=1 / dt_save)
    Zfilt[n, :] = filtfilt(b, a, Zsave[n, :])
Zfilt = Zfilt[:, :-500]

plt.figure()
plt.subplot(8, 1, (1, 7))
plt.plot(np.arange(0, (len(Zfilt) - 1) * dt_save, dt_save), (np.arange(1, N + 1) * np.ones_like(Zfilt)) + (np.real(Zfilt) / (2 * np.std(Zsave, axis=1)[:, None])), 'k')
plt.xlabel('Time (seconds)', fontsize=18)
plt.box(False)
plt.axis('tight')
plt.gca().set_yticks(np.arange(1, N + 1))
plt.gca().set_yticklabels([])
plt.gca().set_yticklabels(label90[::-1, :], fontsize=6)
plt.title(f'Simulated signals in {N} units coupled with K={K} and MD={MD * 1000}ms, filtered between {high_pass}-{low_pass}Hz',
            fontsize=10)
plt.ylim(0, N + 1)
plt.xlim(0, (len(Zfilt) - 1) * dt_save)

plt.subplot(8, 1, 8)
OP = np.abs(np.mean(np.exp(1j * np.angle(hilbert(Zfilt.T))), axis=1))
plt.plot(np.arange(0, (len(OP) - 1) * dt_save, dt_save), OP)
plt.ylim(0, 1)
plt.xlabel('Time (seconds)')
plt.ylabel('Synchrony')
plt.box(False)

plt.show()
                    