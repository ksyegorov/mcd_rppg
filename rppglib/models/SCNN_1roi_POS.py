import math
import os

from scipy import signal
from scipy import sparse
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

NUM_EPCHS = 100
LR = 0.001

class SCNN_1roi_POS:
    def __init__(self, config):
        self.config = config
        self.require_train = True
        self.net = NewPyramidModel_1roi()
        self.device = config.device
        self.weights_name = f'{config.results_folder}/{config.model}__{config.train_dataset}__{config.valid_fold}.pt'
        if os.path.isfile(self.weights_name):
            self.net.load_state_dict(torch.load(self.weights_name))
            

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

        self.net = NewPyramidModel_1roi()
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
        RGB = []
        for frame in video:
            summation = np.sum(np.sum(frame, axis=0), axis=0)
            RGB.append(summation / (frame.shape[0] * frame.shape[1]))
        RGB =  np.asarray(RGB).astype('float32')

        pos_ppg = POS_WANG(RGB, 30)[:, None]
        #print(RGB.shape, pos_ppg.shape)
        RGB = np.concatenate([RGB, pos_ppg], axis=1).astype('float32')   
        return RGB



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

class NewPyramidModel_1roi(torch.nn.Module):
    def __init__(self, ):
        super().__init__()

        in_channels = 4
        
        self.conv1 = torch.nn.Conv1d(in_channels, 128, 7, padding=3)
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.act1 = torch.nn.ReLU()

        self.conv1_2 = torch.nn.Conv1d(128, 128, 5, padding=2)
        self.bn1_2 = torch.nn.BatchNorm1d(128)
        self.act1_2 = torch.nn.ReLU()
        

        self.conv2 = torch.nn.Conv1d(128, 128, 5, padding=2)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.act2 = torch.nn.ReLU()

        self.conv2_2 = torch.nn.Conv1d(128, 128, 3, padding=1)
        self.bn2_2 = torch.nn.BatchNorm1d(128)
        self.act2_2 = torch.nn.ReLU()

        self.conv3 = torch.nn.Conv1d(128, 128, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm1d(128)
        self.act3 = torch.nn.ReLU()

        self.conv3_2 = torch.nn.Conv1d(128, 128, 3, padding=1)
        self.bn3_2 = torch.nn.BatchNorm1d(128)
        self.act3_2 = torch.nn.ReLU()
        
        self.conv4 = torch.nn.Conv1d(128, 128, 3, padding=1)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.act4 = torch.nn.ReLU()

        self.conv4_2 = torch.nn.Conv1d(128, 128, 3, padding=1)
        self.bn4_2 = torch.nn.BatchNorm1d(128)
        self.act4_2 = torch.nn.ReLU()


        self.conv5 = torch.nn.Conv1d(512 + in_channels, 256, 3, padding=1)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.act5 = torch.nn.ReLU()

        self.conv6 = torch.nn.Conv1d(256, 256, 3, padding=1)
        self.bn6 = torch.nn.BatchNorm1d(256)
        self.act6 = torch.nn.ReLU()

        
        self.out_conv = torch.nn.Conv1d(256, 1, 1)
        
    def forward(self, x):
        #print(x.shape)
        x = torch.permute(x, (0, 2, 1))
        #print(x.shape)
        x1 = x

        
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.act1(x1)
        
        x1 = self.conv1_2(x1)
        x1 = self.bn1_2(x1)
        x1 = self.act1_2(x1)

        

        x2 = x1
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.act2(x2)


        x2 = self.conv2_2(x2)
        x2 = self.bn2_2(x2)
        x2 = self.act2_2(x2)

        x3 = x2
        x3 = self.conv3(x3)
        x3 = self.bn3(x3)
        x3 = self.act3(x3)


        x3 = self.conv3_2(x3)
        x3 = self.bn3_2(x3)
        x3= self.act3_2(x3)

        x4 = x3
        x4 = self.conv4(x4)
        x4 = self.bn4(x4)
        x4 = self.act4(x4)

        x4 = self.conv4_2(x4)
        x4 = self.bn4_2(x4)
        x4 = self.act4_2(x4)

        x = torch.cat([x, x1, x2, x3, x4], dim=1)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.act5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.act6(x)

        
        x = self.out_conv(x)
        
        ppg = x[:, 0, :]
        
        mean = ppg.mean(dim=1)
        std = ppg.std(dim=1)
        ppg = (ppg - mean[:, None]) / (std[:, None] + 1e-9)
        return ppg

import numpy as np
from scipy import signal

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
    #BVP = detrend(np.mat(BVP).H, 100)
    BVP = BVP.T
    BVP = np.asarray(np.transpose(BVP))[0]
    b, a = signal.butter(1, [0.75 / fs * 2, 3 / fs * 2], btype='bandpass')
    BVP = signal.filtfilt(b, a, BVP.astype(np.double))

    BVP -= BVP.mean()
    BVP /= BVP.std() + 1e-9
    return BVP