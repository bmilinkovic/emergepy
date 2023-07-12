'''
This script was _altered_ and _edited_ from Michael Schartner's work.

This module defines functions LZc(X), LZs(x), PSpec(X), ACE(X), and SCE(X)
for inputs of multivariate and single-variate EEG or MEG data _X_ (channels x observations).

There are functions supplied which are run by the functions stated above.
For simplicity of use, on would just load this module into their script:

i.e.: from base_directory import lzc
And then use the key functions on our Raw multivariate time-series _X_

'''


from numpy import *
from numpy.linalg import *
from scipy import signal
from scipy.stats import ranksums
from scipy.io import savemat
from scipy.io import loadmat
from random import *
from itertools import combinations

from scipy import signal
from scipy.signal import (butter, lfilter, hilbert, resample)
from pylab import *
import os as os


def Pre2(X):
    '''
    Linear-detrend and subtract mean of X, a multidimensional times series (channels x observations)
    '''
    ro, co = shape(X)
    Z = zeros((ro, co))
    for i in range(ro):  # maybe divide by std?
        Z[i, :] = signal.detrend((X[i, :] - mean(X[i, :])) / std(X[i, :]), axis=0)
    return Z


##############
'''
PSpec; compute spectral power density in canonical EEG bands
'''


##############

def PSpec(X):
    '''
    X: multidimensional time series, ch x obs
    fs: sampling rate in Hz
    '''

    def find_closest(A, target):
        '''
        helper function
        '''
        # A must be sorted
        idx = A.searchsorted(target)
        idx = np.clip(idx, 1, len(A) - 1)
        left = A[idx - 1]
        right = A[idx]
        idx -= target - left < right - target
        return idx

    fs = 250

    de = [1, 4]  # in Hz
    th = [4, 8]
    al = [8, 13]
    be = [13, 30]
    # ga=[30,60]
    # hga=[60,120]

    F = [de, th, al, be]  # ,ga,hga]

    ro, co = shape(X)
    Q = []

    for i in range(ro):

        v = X[i]
        co = len(v)
        N = co  # Number of samplepoints
        T = 1.0 / fs  # sample spacing (denominator in Hz)
        y = v
        yf = fft(y)
        xf = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))
        yff = 2.0 / N * np.abs(yf[0:int(N / 2)])
        bands = zeros(len(F))
        for i in range(len(F)):
            bands[i] = sum(yff[find_closest(xf, F[i][0]):find_closest(xf, F[i][1])])
        bands = bands / sum(bands)
        Q.append(bands)
    return Q


#############
'''
frequency filter
'''


#############

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass(lowcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='highpass')
    return b, a


def butter_highpass_filter(data, lowcut, fs, order):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def notch_iir(fs, f0, data):
    '''
    fs: Sample frequency (Hz)
    f0: Frequency to be removed from signal (Hz)
    '''

    Q = 10.  # 30.0  # Quality factor
    w0 = float(f0) / (fs / 2)  # Normalized Frequency
    b, a = signal.iirnotch(w0, Q)
    return lfilter(b, a, data)


##########
'''
LZc - Lempel-Ziv Complexity, column-by-column concatenation

X is continuous multidimensional time series, channels x observations
'''


##########

def cpr(string):
    '''
    Lempel-Ziv-Welch compression of binary input string, e.g. string='0010101'. It outputs the size of the dictionary of binary words.
    '''
    d = {}
    w = ''
    for c in string:
        wc = w + c
        if wc in d:
            w = wc
        else:
            d[wc] = wc
            w = c
    return len(d)


def str_col(X):
    '''
    Input: Continuous multidimensional time series
    Output: One string being the binarized input matrix concatenated comlumn-by-column
    '''
    ro, co = shape(X)
    TH = zeros(ro)
    M = zeros((ro, co))
    for i in range(ro):
        M[i, :] = abs(hilbert(X[i, :]))
        TH[i] = mean(M[i, :])

    s = ''
    for j in range(co):
        for i in range(ro):
            if M[i, j] > TH[i]:
                s += '1'
            else:
                s += '0'

    return s


def LZc(X):
    '''
    Compute LZc and use shuffled result as normalization
    '''
    X = Pre2(X)
    SC = str_col(X)
    M = list(SC)
    shuffle(M)
    w = ''
    for i in range(len(M)):
        w += M[i]
    return cpr(SC) / float(cpr(w))


def LZs(x):
    '''
    Lempel ziv complexity of single timeseries
    '''

    co = len(x)
    x = signal.detrend((x - mean(x)) / std(x), axis=0)
    s = ''
    r = abs(hilbert(x))
    th = mean(r)

    for j in range(co):
        if r[j] > th:
            s += '1'
        else:
            s += '0'

    M = list(s)
    shuffle(M)
    w = ''
    for i in range(len(M)):
        w += M[i]

    return cpr(s) / float(cpr(w))


##########
'''
ACE - Amplitude Coalition Entropy
'''


##########

def map2(psi):
    '''
    Bijection, mapping each binary column of binary matrix psi onto an integer.
    '''
    ro, co = shape(psi)
    c = zeros(co)
    for t in range(co):
        for j in range(ro):
            c[t] = c[t] + psi[j, t] * (2 ** j)
    return c


def binTrowH(M):
    '''
    Input: Multidimensional time series M
    Output: Binarized multidimensional time series
    '''
    ro, co = shape(M)
    M2 = zeros((ro, co))
    for i in range(ro):
        M2[i, :] = signal.detrend(M[i, :], axis=0)
        M2[i, :] = M2[i, :] - mean(M2[i, :])
    M3 = zeros((ro, co))
    for i in range(ro):
        M2[i, :] = abs(hilbert(M2[i, :]))
        th = mean(M2[i, :])
        for j in range(co):
            if M2[i, j] >= th:
                M3[i, j] = 1
            else:
                M3[i, j] = 0
    return M3


def entropy(string):
    '''
    Calculates the Shannon entropy of a string
    '''
    string = list(string)
    prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
    entropy = - sum([p * log(p) / log(2.0) for p in prob])

    return entropy


def ACE(X):
    '''
    Measure ACE, using shuffled result as normalization.
    '''
    X = Pre(X)
    ro, co = shape(X)
    M = binTrowH(X)
    E = entropy(map2(M))
    for i in range(ro):
        shuffle(M[i])
    Es = entropy(map2(M))
    return E / float(Es)


##########
'''
SCE - Synchrony Coalition Entropy
'''


##########

def diff2(p1, p2):
    '''
    Input: two series of phases
    Output: synchrony time series thereof
    '''
    d = array(abs(p1 - p2))
    d2 = zeros(len(d))
    for i in range(len(d)):
        if d[i] > pi:
            d[i] = 2 * pi - d[i]
        if d[i] < 0.8:
            d2[i] = 1

    return d2


def Psi(X):
    '''
    Input: Multi-dimensional time series X
    Output: Binary matrices of synchrony for each series
    '''
    X = angle(hilbert(X))
    ro, co = shape(X)
    M = zeros((ro, ro - 1, co))

    # An array containing 'ro' arrays of shape 'ro' x 'co', i.e. being the array of synchrony series for each channel.

    for i in range(ro):
        l = 0
        for j in range(ro):
            if i != j:
                M[i, l] = diff2(X[i], X[j])
                l += 1

    return M


def BinRan(ro, co):
    '''
    Create random binary matrix for normalization
    '''

    y = rand(ro, co)
    for i in range(ro):
        for j in range(co):
            if y[i, j] > 0.5:
                y[i, j] = 1
            else:
                y[i, j] = 0
    return y


def SCE(X):
    X = Pre(X)
    ro, co = shape(X)
    M = Psi(X)
    ce = zeros(ro)
    norm = entropy(map2(BinRan(ro - 1, co)))
    for i in range(ro):
        c = map2(M[i])
        ce[i] = entropy(c)

    return mean(ce) / norm, ce / norm
