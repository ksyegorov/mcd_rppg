import numpy as np
from scipy import signal
import math

import scipy
from scipy.signal import butter
from scipy.sparse import spdiags
from scipy import linalg
from scipy import signal
from scipy import sparse

def pos(rgb, framerate):
    seg_t = 3.2
    l = int(framerate * seg_t)
    H = np.zeros(rgb.shape[0])    
    B = [int(0.8 // (framerate / l)), int(4 // (framerate / l))]    
    for t in range(0, (rgb.shape[0] - l + 1)):
        C = rgb[t:t+l,:].T
        mean_color = np.mean(C, axis=1)
        diag_mean_color = np.diag(mean_color)
        diag_mean_color_inv = np.linalg.inv(diag_mean_color)
        Cn = np.matmul(diag_mean_color_inv, C)
        projection_matrix = np.array([[0,1,-1],[-2,1,1]])
        S = np.matmul(projection_matrix, Cn)
        std = np.array([1, np.std(S[0, :]) / np.std(S[1, :])])
        P = np.matmul(std, S)
        H[t: t+l] = H[t: t+l] +  (P - np.mean(P))
    return H


def chrome(rgb, framerate):
    LPF = 0.7
    HPF = 2.5
    WinSec = 1.6

    FN = rgb.shape[0]
    NyquistF = 1 / 2 * framerate
    B, A = signal.butter(3, [LPF/NyquistF, HPF/NyquistF], 'bandpass')

    WinL = math.ceil(WinSec * framerate)
    if(WinL % 2):
        WinL = WinL+1
    NWin = math.floor((FN-WinL//2)/(WinL//2))
    WinS = 0
    WinM = int(WinS+WinL//2)
    WinE = WinS+WinL
    totallen = (WinL//2)*(NWin+1)
    S = np.zeros(totallen)

    for i in range(NWin):
        RGBBase = np.mean(rgb[WinS:WinE, :], axis=0)
        RGBNorm = np.zeros((WinE-WinS, 3))
        for temp in range(WinS, WinE):
            RGBNorm[temp-WinS] = np.true_divide(rgb[temp], RGBBase)
        Xs = np.squeeze(3*RGBNorm[:, 0]-2*RGBNorm[:, 1])
        Ys = np.squeeze(1.5*RGBNorm[:, 0]+RGBNorm[:, 1]-1.5*RGBNorm[:, 2])
        Xf = signal.filtfilt(B, A, Xs, axis=0)
        Yf = signal.filtfilt(B, A, Ys)

        Alpha = np.std(Xf) / np.std(Yf)
        SWin = Xf-Alpha*Yf
        SWin = np.multiply(SWin, signal.windows.hann(WinL))

        temp = SWin[:int(WinL//2)]
        S[WinS:WinM] = S[WinS:WinM] + SWin[:int(WinL//2)]
        S[WinM:WinE] = SWin[int(WinL//2):]
        WinS = WinM
        WinM = WinS+WinL//2
        WinE = WinS+WinL
    BVP = S
    return BVP


def POS_WANG(RGB, fs):
    WinSec = 1.6
        
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)

    for n in range(N):
        m = n - l
        if m >= 0:
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.mat(Cn).H
            S = np.matmul(np.array([[0, 1, -1], [-2, 1, 1]]), Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h
            H[0, m:n] = H[0, m:n] + (h[0])

    BVP = H
    BVP = detrend(np.mat(BVP).H, 100)
    BVP = np.asarray(np.transpose(BVP))[0]
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))
    return BVP


def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal