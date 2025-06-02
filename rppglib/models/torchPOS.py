import numba
import numpy as np
import torch
import torchaudio
import math
from scipy import signal

@numba.njit(parallel=False) # on some processors this is faster #@numba.njit(parallel=True)
def extract_mean_frame(frame):
    H, W, Ch = frame.shape
    means = np.empty(Ch, dtype=np.float32)
    for ch in range(Ch):
        ch_sum = 0
        ch_count = 0
        for i in numba.prange(H):
            for j in numba.prange(W):
                val = frame[i, j, ch]
                #if val != 0:
                ch_sum += val
                ch_count += 1
        means[ch] = ch_sum / (ch_count + 1e-6)        
    return means

@numba.njit(parallel=True) # on some processors this is faster #@numba.njit(parallel=True)
def extract_mean_video(video):
    rgbs = np.empty((video.shape[0], 3), dtype=np.float32)
    for i in numba.prange(video.shape[0]):
        frame = video[i]
        rgb = extract_mean_frame(frame)
        rgbs[i, :] = rgb
    return rgbs


class torchPOS(torch.nn.Module):
    def __init__(self, fs=30, window_sec=1.6, low_freq=0.75, high_freq=3.0):
        super().__init__()
        self.matrix = torch.nn.Parameter(data=torch.Tensor([[0, 1, -1], [-2, 1, 1], [-1, 2, -1]]).float(), requires_grad=False)
        self.window = math.ceil(window_sec * fs)
        self.fs = fs
        
        b, a = signal.butter(1, [low_freq / fs * 2, high_freq / fs * 2], btype='bandpass')
        self.a = torch.nn.Parameter(torch.Tensor(a), requires_grad=False)
        self.b = torch.nn.Parameter(torch.Tensor(b), requires_grad=False)

    def filter(self, ppg):
        ppg = torchaudio.functional.filtfilt(ppg, self.a, self.b)
        return ppg

    def window_function(self, rgb):
        Cn = self.normalize_rgb(rgb)
        S = torch.matmul(self.matrix, Cn)
        S_0 = S[:, :, 0, :]
        S_1 = S[:, :, 1, :]
        S_3 = S[:, :, 2, :]
    
        std_div = S_0.std(dim=2) / (S_1.std(dim=2) + 1e-4)
        std_div = std_div[:, :, None]
        h = S_0 + std_div * S_1 
        return h

    def normalize_rgb(self, rgb):
        means = rgb.mean(dim=3)
        rgb = rgb / (means[:, :, :, None] + 1e-6)
        return rgb

    def forward(self, videos):
        rgb = self.process_videos(videos)
        ppg = self.rgb_to_ppg(rgb)
        return ppg

    def rgb_to_ppg(self, rgb):
        batch_size, time, channel = rgb.shape
        unfolded = rgb.unfold(1, self.window, 1)
        hs = self.window_function(unfolded)
        total_length, batch_size = rgb.shape[1], rgb.shape[0]
        ppg = torch.zeros((batch_size, total_length), dtype=torch.float32).to(rgb.device)
        for i, end in enumerate(range(self.window, total_length)):
            start = end - self.window
            ppg[:, start:end] += hs[:, i] 
        ppg = self.filter(ppg)
        return ppg        

    def process_videos(self, videos):
        result = list()
        for video in tqdm(videos):
            rgbs = extract_mean_video(video.numpy())
            rgbs = torch.from_numpy(rgbs)                    
            result.append(rgbs)
        result = torch.stack(result)
        return result