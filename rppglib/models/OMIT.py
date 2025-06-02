"""OMIT
Face2PPG: An unsupervised pipeline for blood volume pulse extraction from faces.
Álvarez Casado, C., & Bordallo López, M.
IEEE Journal of Biomedical and Health Informatics.
(2023).
"""

import numpy as np
import torch
import scipy
import scipy.signal
#from unsupervised_methods import utils


class OMIT:
    def __init__(self, config):
        self.config = config
        self.require_train = False

    def predict(self, videos):
        videos_np = videos.detach().cpu().numpy()
        preds = list()
        for video in videos_np:
            assert not np.isnan(videos).any(), 'NaN in video'
            pred_ppg = OMIT_cpu(video)
            assert not np.isnan(pred_ppg).any(), 'NaN in pred PPG'
            preds.append(pred_ppg)
        preds = np.stack(preds).astype('float32')
        preds = torch.from_numpy(preds).to(videos.device)
        return preds

    @staticmethod
    def video_processing(frames, landmarks):
        RGB = []
        for frame in frames:
            summation = np.sum(np.sum(frame, axis=0), axis=0)
            RGB.append(summation / (frame.shape[0] * frame.shape[1]))
        RGB = np.asarray(RGB)
        RGB = RGB.transpose(1, 0).reshape(1, 3, -1)
        return np.asarray(RGB)



def OMIT_cpu(frames):
    precessed_data = frames[0]
    Q, R = np.linalg.qr(precessed_data)
    S = Q[:, 0].reshape(1, -1)
    P = np.identity(3) - np.matmul(S.T, S)
    Y = np.dot(P, precessed_data)
    bvp = Y[1, :]
    bvp = bvp.reshape(-1)
    sos = scipy.signal.cheby2(N=4,  # Filter order
                    rs=30,  # Stop-band attenuation (dB)
                    Wn=[0.75, 3],  # Passband frequencies [low, high] (Hz)
                    btype='bandpass',  # Bandpass filter
                    fs=30,  # Sampling frequency
                    output='sos') # Second-order sections (more stable)
    bvp = scipy.signal.sosfilt(sos, bvp)
    return bvp