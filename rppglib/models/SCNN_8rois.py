import math
import os

from scipy import signal
from scipy import sparse
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import numba
import cv2
import hashlib

from . import torchPOS

NUM_EPCHS = 50
LR = 0.001


# FACE_SEG = np.array([338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10])

# FOREHEAD = np.array([10, 338, 297, 332, 333, 334, 296, 336, 9, 107, 66, 105, 63, 68, 54, 103, 67, 109])
# LOW_FOREHEAD = np.array([151, 337, 299, 333, 334, 296, 336, 9, 107, 66, 105, 104, 69, 108])
# NOSE = np.array([168, 417, 351, 419, 248, 281, 363, 344, 438, 457, 274, 354, 19, 125, 44, 237, 218, 115, 134, 51, 3, 196, 122, 193])
# UPPER_LEFT_CHEEK  = np.array([116, 117, 118, 119, 120, 100, 142, 203, 206, 207, 147, 123])
# UPPER_RIGHT_CHEEK = np.array([345, 346, 347, 348, 349, 329, 371, 423, 426, 427, 376, 352])

FACE = np.array([1, 2, 3, 4, 5, 6, 8, 9, 10, 11,12,13,14,15,16,17,27,26,25,24,23,22,21,20,19,18])
NOSE = np.array([22, 23, 43, 36, 35, 34,33,32,40])
LEFT_UPPER_CHEEK = np.array([1, 37, 42, 41,40,32,3,2])
RIGHT_UPPER_CHEEK = np.array([17, 46, 47, 48, 43,36,15,16])
LEFT_LOWER_CHEEK = np.array([3, 32, 51, 50, 49, 5, 4])
RIGHT_LOWER_CHEEK = np.array([15, 36, 53, 54, 55, 13, 14])
LIPS = np.array([49,50,51,52,53,54,55,56,57,58,59,60])
CHIN = np.array([5, 6, 7, 8, 9, 10, 11,12,13,56,57,58,59,60])

ALL_REGIONS = [FACE, NOSE, LEFT_UPPER_CHEEK, RIGHT_UPPER_CHEEK, LEFT_LOWER_CHEEK, RIGHT_LOWER_CHEEK, LIPS, CHIN]


def process_frame(frame, face):
    points, coords = face
    mask = get_mask(frame, points)
    face = frame * mask    
    face_means = extract_mean(face)
    return face_means


def get_mask(frame, points, seg=None):
    points = np.array(points)
    mask = np.zeros_like(frame)
    pps = points[seg].astype('int32')
    #print(pps.dtype, pps.shape, frame.shape)
    polys = [pps, ]
    #print(polys)
    cv2.fillPoly(mask, polys, (255, 255, 255))
    mask = mask > 125
    return mask


@numba.njit(parallel=False) # on some processors this is faster #@numba.njit(parallel=True)
def extract_mean(arr):
    H, W, Ch = arr.shape
    means = np.empty(Ch, dtype=np.float32)
    for ch in range(Ch):
        ch_sum = 0
        ch_count = 0
        for i in numba.prange(H):
            for j in numba.prange(W):
                val = arr[i, j, ch]
                if val != 0:
                    ch_sum += val
                    ch_count += 1
        means[ch] = ch_sum / (ch_count + 1e-6)        
    return means



class SCNN_8rois:
    def __init__(self, config):
        self.config = config
        self.require_train = True
        self.net = SCNN_8roi_NN()
        self.device = config.device
        self.weights_name = f'{config.results_folder}/{config.model}__{config.train_dataset}__{config.valid_fold}.pt'
        if os.path.isfile(self.weights_name):
            pass
            #self.net.load_state_dict(torch.load(self.weights_name))
            

    def predict(self, videos):
        videos = videos.cpu()
        self.net.cpu()
        self.net.eval()
        with torch.no_grad():
            preds = self.net(videos)
        return preds

    def train(self, train_dl, valid_dl):
        net = self.net
        net.to(self.device)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)  
        criterion = torch.nn.MSELoss()

        history = list()
        
        for i in tqdm(range(NUM_EPCHS), desc='Training epochs'):
            
            net.train()
            train_loss = 0
            for videos, true_ppgs in train_dl:
                
                videos = videos.to(self.device)
                true_ppgs = true_ppgs.to(self.device)
                
                assert not torch.isnan(videos).any(), 'NaNs in Videos'
                assert not torch.isnan(true_ppgs).any(), 'NaNs in PPGs'

                pred_ppgs = net(videos)
                assert not torch.isnan(pred_ppgs).any(), 'NaNs in pred PPGs'
                
                loss = criterion(pred_ppgs, true_ppgs)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()   

                train_loss += loss.item()

            net.eval()
            valid_loss = 0
            best_valid_loss = 1e10
            with torch.no_grad():
                for videos, true_ppgs in valid_dl:
                    
                    videos = videos.to(self.device)
                    true_ppgs = true_ppgs.to(self.device)
                    
                    assert not torch.isnan(videos).any(), 'NaNs in Videos'
                    assert not torch.isnan(true_ppgs).any(), 'NaNs in PPGs'
    
                    pred_ppgs = net(videos)
                    assert not torch.isnan(pred_ppgs).any(), 'NaNs in pred PPGs'
                    
                    loss = criterion(pred_ppgs, true_ppgs)
    
                    valid_loss += loss.item()  
                    
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(net.state_dict(), self.weights_name)

            history.append({'train_loss': train_loss, 'valid_loss': valid_loss})

        self.net = SCNN_8roi_NN()
        self.net.load_state_dict(torch.load(self.weights_name))

        history = pd.DataFrame(history)
        print('Training process:')
        plt.figure(figsize=(20, 3))
        plt.plot(history['train_loss'])
        plt.plot(history['valid_loss'])
        plt.grid()
        plt.show()


    
    @staticmethod
    def video_processing(video, landmarks):
        video_hash = str(hashlib.md5(video.tobytes()).hexdigest())[:8]
        landmarks_hash = str(hashlib.md5(landmarks.tobytes()).hexdigest())[:8]
        hash_file = f'cache/{video_hash}_{landmarks_hash}.npy'

        if not os.path.isfile(hash_file):
            pos = torchPOS.torchPOS()
            RGB = []
            for seg in ALL_REGIONS:
                seg_rgbs = list()
                for frame, lms in zip(video, landmarks):
                    mask = get_mask(frame, lms, seg=seg)
                    masked = frame * mask
                    seg_rgb = extract_mean(masked)
                    seg_rgbs.append(seg_rgb)
                seg_rgbs = np.asarray(seg_rgbs)
                ppg = pos.rgb_to_ppg(torch.from_numpy(seg_rgbs)[None, :, :]).numpy()[0]
                ppg = ppg[:, None]
                seg_rgbs = np.concatenate([seg_rgbs, ppg], axis=1)
                RGB.append(seg_rgbs)
            RGB = np.concatenate(RGB, axis=1)
            np.save(hash_file, RGB)
        return np.load(hash_file)




def net_train(net, dl, optimizer, criterion, device='cpu'):
    net.to(device)
    net.train()
    
    epoch_maes = {target: 0.0 for target in net.params['targets']}
    epoch_maes['hr'] = 0.0
    mae_loss = torch.nn.L1Loss()
    
    for batch in dl:

        rgb = batch['rgb'].to(device)
        assert not torch.isnan(rgb).any(), 'NaNs in RGB'

        pred_ppg, other_targets = net(rgb)
        assert not torch.isnan(pred_ppg).any(), 'NaNs in output (no NaNs in RGB)'
        assert not torch.isnan(other_targets).any(), 'NaNs in AP (no NaNs in RGB)'

        true_ppg = batch['ppg'].to(device)
        assert not torch.isnan(true_ppg).any()

        loss = criterion(pred_ppg, true_ppg)

        for i, target in enumerate(net.params['targets']):
            true = batch[target].to(device)
            pred = other_targets[:, i]
            ml = mae_loss(pred, true) 
            loss += ml
            epoch_maes[target] += ml.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_maes['hr'] += calc_hr_diffs(pred_ppg, true_ppg)
        
    epoch_loss /= (len(dl) + 1e-9)
    epoch_maes = {key: val / (len(dl) + 1e-9) for key, val in epoch_maes.items()}
    epoch_maes['loss'] = epoch_loss
    return epoch_maes



class Conv2dBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=None, padding=None):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel, stride=stride, bias=False, padding=padding)
        self.bn = torch.nn.BatchNorm2d(out_channels)
        self.act = torch.nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

class Conv1dBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=None, padding=None):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel, stride=stride, bias=False, padding=padding)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.act = torch.nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class InvBottleneckBlock1d(torch.nn.Module):
    def __init__(self, channels, kernel, mult=4):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(channels, channels*mult, kernel, stride=1, bias=False, padding=kernel//2)
        self.norm1 = torch.nn.BatchNorm1d(channels*mult)
        self.act1 = torch.nn.GELU()
        self.conv2 = torch.nn.Conv1d(channels*mult, channels, kernel, stride=1, bias=False, padding=kernel//2)
        self.norm2 = torch.nn.BatchNorm1d(channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class ResBlock1d(torch.nn.Module):
    def __init__(self, channels, kernel, mult=4):
        super().__init__()
        self.block = InvBottleneckBlock1d(channels, kernel, mult=mult)
        self.act = torch.nn.GELU()

    def forward(self, x):
        x = x + self.block(x)
        x = self.act(x)
        return x
        

class Conv1dBlockRes(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=None, padding=None):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel, stride=stride, bias=False, padding=padding)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.act = torch.nn.GELU()
        self.res_block = ResBlock1d(out_channels, kernel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.res_block(x)
        return x        


class SCNN_8roi_NN(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

        channels = 32

        self.in_conv_1 = Conv2dBlock(1,          channels,   (4, 5), stride=(4, 1), padding=(0, 2))
        self.in_conv_2 = Conv2dBlock(channels,   channels*2, (1, 5), stride=(1, 1), padding=(0, 2))
        self.in_conv_3 = Conv2dBlock(channels*2, channels*2, (1, 5), stride=(1, 1), padding=(0, 2))
        self.in_conv_4 = Conv2dBlock(channels*2, channels*2, (1, 5), stride=(1, 1), padding=(0, 2))

        self.attention = torch.nn.Conv2d(channels*2, 1, 1, bias=False)
        self.attention_act = torch.nn.Softmax(dim=2)

        self.out_conv_1 = Conv1dBlockRes(channels*2, channels*2, 5, stride=1, padding=2)
        self.out_conv_2 = Conv1dBlockRes(channels*2, channels*2, 5, stride=1, padding=2)
        self.out_conv_3 = Conv1dBlockRes(channels*2, channels*2, 5, stride=1, padding=2)
        self.out_conv_4 = Conv1dBlockRes(channels*2, channels*2, 5, stride=1, padding=2)

        self.out_conv_5 = torch.nn.Conv1d(channels*2, 1, 1)
        
    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))
        x = x[:, None, :, :]
        x = self.in_conv_1(x)
        x = self.in_conv_2(x)
        x = self.in_conv_3(x)
        x = self.in_conv_4(x)
        att = self.attention(x)
        att = self.attention_act(att)
        x = x * att
        x = x.sum(dim=2)
        x = self.out_conv_1(x)
        x = self.out_conv_2(x)
        x = self.out_conv_3(x)
        x = self.out_conv_4(x)

        x = self.out_conv_5(x)

        ppg = x[:, 0, :]
        
        # mean = ppg.mean(dim=1)
        # std = ppg.std(dim=1)
        # ppg = (ppg - mean[:, None]) / (std[:, None] + 1e-9)
        return ppg