import torch
import cv2
import numpy as np
import functools
import scipy.signal
from tqdm import tqdm

class RPPGDataset(torch.utils.data.Dataset):
    def __init__(self, video_files, ppg_files, config, train=False):
        assert len(video_files) == len(ppg_files)
        self.video_files = video_files
        self.ppg_files = ppg_files
        self.window = config['window']
        self.config = config
        self.train = train

        self.videos = [self.open_and_preprocess_video(video_file) for video_file in tqdm(self.video_files)]
        self.ppgs = [self.open_and_preprocess_ppg(video_file) for video_file in tqdm(self.ppg_files)]

    def __len__(self, ):
        return len(self.video_files) * self.config['samples_per_video']

    def __getitem__(self, index):
        index = index % len(self.video_files)
        #video_file = self.video_files[index]
        #ppg_file = self.ppg_files[index]

        #video = self.open_and_preprocess_video(video_file)
        #ppg = self.open_and_preprocess_ppg(ppg_file)

        video = self.videos[index]
        ppg = self.ppgs[index]

        assert len(video.shape) == 4
        assert len(ppg.shape) == 1
        assert video.shape[0] == ppg.shape[0]
        assert video.shape[3] == 3

        video, ppg = self.select_window(video, ppg)
        video = self.preprocess_video_window(video)
        ppg = self.preprocess_ppg_window(ppg)

        output = dict()
        output['video'] = video
        output['ppg'] = ppg
        return output

    def select_window(self, video, ppg):
        if ppg.shape[0] < self.window:
            video, ppg = self._zero_pad(video, ppg)
        elif ppg.shape[0] > self.window:
            video, ppg = self._random_crop(video, ppg)
        video = video.copy()
        ppg = ppg.copy()
        return video, ppg

    def _zero_pad(self, video, ppg):
        assert video.shape[0] == ppg.shape[0]
        assert video.shape[0] < self.window

        orig_size = ppg.shape[0]
        H = video.shape[1]
        W = video.shape[2]
        
        padded_ppg = np.zeros(self.window, dtype=ppg.dtype)
        padded_ppg[:orig_size] = ppg
        padded_ppg[orig_size:] = ppg.mean()

        padded_video = np.zeros((self.window, H, W, 3), dtype=video.dtype)
        padded_video[:orig_size] = video
        padded_video[orig_size:] = video.mean(axis=0)
        return padded_video, padded_ppg

    def _random_crop(self, video, ppg):
        assert video.shape[0] == ppg.shape[0]
        assert video.shape[0] > self.window

        start = np.random.randint(ppg.shape[0] - self.window)
        end = start + self.window
        return video[start: end], ppg[start: end]

    def open_and_preprocess_video(self, video_file):
        raise NotImplementedError
        
    def open_and_preprocess_ppg(self, ppg_file):
        raise NotImplementedError 

    
    def preprocess_video_window(self, video):
        return video

    def preprocess_ppg_window(self, ppg):
        ppg = bandpass_filter(ppg, self.config['sr'], self.config['ppg_low_freq'], self.config['ppg_high_freq'])
        ppg -= ppg.mean()
        ppg /= ppg.std()
        return ppg 

    def to_dl(self, ):
        bs = self.config['batch_size']
        nw = self.config['num_workers']
        train = self.train
        return torch.utils.data.DataLoader(self, batch_size=bs, num_workers=nw, shuffle=train)
        


class UBFC_rPPG_Dataset(RPPGDataset):
    def open_and_preprocess_video(self, video_file):
        return cached_video_load_ubfcrppg(video_file)


    def open_and_preprocess_ppg(self, ppg_file):
        return cached_ppg_load_ubfcrppg(ppg_file)       


#@functools.lru_cache
def cached_video_load_ubfcrppg(video_file):
    """Reads a video file, returns frames(T, H, W, 3) """
    VidObj = cv2.VideoCapture(video_file)
    VidObj.set(cv2.CAP_PROP_POS_MSEC, 0)
    success, frame = VidObj.read()
    frames = list()
    while success:
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
        frame = np.asarray(frame)
        frames.append(frame)
        success, frame = VidObj.read()
    return np.asarray(frames)    

#@functools.lru_cache
def cached_ppg_load_ubfcrppg(ppg_file):
    with open(ppg_file, "r") as f:
        str1 = f.read().strip().split('  ')
    bvp = [float(x.strip()) for x in str1]
    file_len = len(bvp)
    assert file_len % 3 == 0
    bvp = bvp[:file_len//3]
    return np.array(bvp)  

def filter_signal(signal, rate, freq, mode='high', order=4):
    hb_n_freq = freq / (rate / 2)
    b, a = scipy.signal.butter(order, hb_n_freq, mode)
    filtered = scipy.signal.filtfilt(b, a, signal)
    filtered = filtered.astype(signal.dtype)
    return filtered

def bandpass_filter(signal, rate, low_freq, high_freq, order=4):
    signal = filter_signal(signal, rate, high_freq, mode='low', order=order)
    signal = filter_signal(signal, rate, low_freq,  mode='high',  order=order)
    return signal