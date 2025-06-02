import math

from scipy import signal
from scipy import sparse
import numpy as np
import torch

class myPOS_v1:
    def __init__(self, config):
        self.config = config
        self.require_train = False

    def predict(self, videos):
        videos_np = videos.detach().cpu().numpy()
        preds = list()
        for video in videos_np:
            assert not np.isnan(videos).any(), 'NaN in video'
            pred_ppg = POS_WANG(video, self.config.fps)
            assert not np.isnan(pred_ppg).any(), 'NaN in pred PPG'
            preds.append(pred_ppg)
        preds = np.stack(preds).astype('float32')
        preds = torch.from_numpy(preds).to(videos.device)
        return preds

    @staticmethod
    def video_processing(video, landmarks):
        RGB = []
        for frame in video:
            summation = np.sum(np.sum(frame, axis=0), axis=0)
            RGB.append(summation / (frame.shape[0] * frame.shape[1]))
        RGB =  np.asarray(RGB).astype('float32')
        return RGB


def POS_WANG(RGB, fs):
    WinSec = 1.6
    N = RGB.shape[0]
    H = np.zeros((1, N))
    l = math.ceil(WinSec * fs)
    ort_matrix = np.array([[0, 1, -1], [-1.5, 1.5, 0.5]])
    for n in range(N):
        m = n - l
        if m >= 0:
            Cn = np.true_divide(RGB[m:n, :], np.mean(RGB[m:n, :], axis=0))
            Cn = np.mat(Cn).H
            S = np.matmul(ort_matrix, Cn)
            h = S[0, :] + (np.std(S[0, :]) / np.std(S[1, :])) * S[1, :]
            mean_h = np.mean(h)
            for temp in range(h.shape[1]):
                h[0, temp] = h[0, temp] - mean_h
            H[0, m:n] = H[0, m:n] + (h[0])

    BVP = H
    #BVP = detrend(np.mat(BVP).H, 100)
    BVP = BVP.T
    BVP = np.asarray(np.transpose(BVP))[0]
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))

    BVP -= BVP.mean()
    BVP /= BVP.std() + 1e-9
    return BVP


def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = sparse.spdiags(diags_data, diags_index,(signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot((H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal