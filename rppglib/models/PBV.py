import math

from scipy import signal
from scipy import sparse
import numpy as np
import torch

class PBV:
    def __init__(self, config):
        self.config = config
        self.require_train = False

    def predict(self, videos):
        videos_np = videos.detach().cpu().numpy()
        preds = list()
        for video in videos_np:
            assert not np.isnan(videos).any(), 'NaN in video'
            pred_ppg = cpu_PBV(video)
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


def cpu_PBV(signal):
    """
    PBV method on CPU using Numpy.

    De Haan, G., & Van Leest, A. (2014). Improved motion robustness of remote-PPG by using the blood volume pulse signature. Physiological measurement, 35(9), 1913.
    """
    signal = signal.T[None, :, :]
    X = signal
    U, _, _ = np.linalg.svd(X)
    S = U[:, :, 0]
    S = np.expand_dims(S, 2)
    sst = np.matmul(S, np.swapaxes(S, 1, 2))
    p = np.tile(np.identity(3), (S.shape[0], 1, 1))
    P = p - sst
    Y = np.matmul(P, X)
    bvp = Y[0, 1, :]
    return bvp 