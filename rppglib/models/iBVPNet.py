import torch
import numpy as np
import matplotlib.pyplot
from tqdm import tqdm
import rppglib.processing
import pandas as pd
import matplotlib.pyplot as plt
import os

IMG_SIZE = (600, 72, 72)
PATCHES = (4, 4, 4)
NUM_EPCHS = 10
LR = 0.0001


class iBVPNet:
    def __init__(self, config):
        self.config = config
        self.require_train = True
        self.net = iBVPNet_NN(frames=IMG_SIZE[0])
        self.device = config.device
        self.weights_name = f'{config.results_folder}/{config.model}__{config.train_dataset}__{config.valid_fold}.pt'
        if os.path.isfile(self.weights_name):
            self.net.load_state_dict(torch.load(self.weights_name))

    def predict(self, videos):
        videos = videos.to(self.device)
        self.net.to(self.device)
        self.net.eval()
        with torch.no_grad():
            preds = self.net(videos)
        return preds.cpu()

    def train(self, train_dl, valid_dl):
        net = self.net
        net.to(self.device)
        
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)  
        criterion = torch.nn.MSELoss()

        history = list()
        
        for i in range(NUM_EPCHS):
            
            net.train()
            train_loss = 0
            for videos, true_ppgs in tqdm(train_dl, desc=f'Epoch {i} training'):
 
                videos = videos.to(self.device)
                true_ppgs = true_ppgs.to(self.device)[:, :IMG_SIZE[0]]
                
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
                for videos, true_ppgs in tqdm(train_dl, desc=f'Epoch {i} validation'):
                    
                    videos = videos.to(self.device)
                    true_ppgs = true_ppgs.to(self.device)[:, :IMG_SIZE[0]]
                    
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

        self.net = iBVPNet_NN(frames=IMG_SIZE[0])
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
        video = video
        video = rppglib.processing.resize_video(video, IMG_SIZE[1], IMG_SIZE[2])
        video = np.transpose(video, (3, 0, 1, 2)).astype('float32')
        return video


"""iBVPNet - 3D Convolutional Network.
Proposed along with the iBVP Dataset, see https://doi.org/10.3390/electronics13071334

Joshi, Jitesh, and Youngjun Cho. 2024. "iBVP Dataset: RGB-Thermal rPPG Dataset with High Resolution Signal Quality Labels" Electronics 13, no. 7: 1334.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(ConvBlock3D, self).__init__()
        self.conv_block_3d = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size, stride, padding),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.conv_block_3d(x)


class DeConvBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(DeConvBlock3D, self).__init__()
        k_t, k_s1, k_s2 = kernel_size
        s_t, s_s1, s_s2 = stride
        self.deconv_block_3d = nn.Sequential(
            nn.ConvTranspose3d(in_channel, in_channel, (k_t, 1, 1), (s_t, 1, 1), padding),
            nn.Tanh(),
            nn.InstanceNorm3d(in_channel),
            
            nn.Conv3d(in_channel, out_channel, (1, k_s1, k_s2), (1, s_s1, s_s2), padding),
            nn.Tanh(),
            nn.InstanceNorm3d(out_channel),
        )

    def forward(self, x):
        return self.deconv_block_3d(x)

# num_filters
nf = [8, 16, 24, 40, 64]

class encoder_block(nn.Module):
    def __init__(self, in_channel, debug=False):
        super(encoder_block, self).__init__()
        # in_channel, out_channel, kernel_size, stride, padding

        self.debug = debug
        self.spatio_temporal_encoder = nn.Sequential(
            ConvBlock3D(in_channel, nf[0], [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(nf[0], nf[1], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(nf[1], nf[2], [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(nf[2], nf[3], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
            nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2)),
            ConvBlock3D(nf[3], nf[4], [1, 3, 3], [1, 1, 1], [0, 1, 1]),
            ConvBlock3D(nf[4], nf[4], [3, 3, 3], [1, 1, 1], [1, 1, 1]),
        )

        self.temporal_encoder = nn.Sequential(
            ConvBlock3D(nf[4], nf[4], [11, 1, 1], [1, 1, 1], [5, 0, 0]),
            ConvBlock3D(nf[4], nf[4], [11, 3, 3], [1, 1, 1], [5, 1, 1]),
            nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2)),
            ConvBlock3D(nf[4], nf[4], [11, 1, 1], [1, 1, 1], [5, 0, 0]),
            ConvBlock3D(nf[4], nf[4], [11, 3, 3], [1, 1, 1], [5, 1, 1]),
            nn.MaxPool3d((2, 2, 2), stride=(2, 1, 1)),
            ConvBlock3D(nf[4], nf[4], [7, 1, 1], [1, 1, 1], [3, 0, 0]),
            ConvBlock3D(nf[4], nf[4], [7, 3, 3], [1, 1, 1], [3, 1, 1])
        )

    def forward(self, x):
        if self.debug:
            print("Encoder")
            print("x.shape", x.shape)
        st_x = self.spatio_temporal_encoder(x)
        if self.debug:
            print("st_x.shape", st_x.shape)
        t_x = self.temporal_encoder(st_x)
        if self.debug:
            print("t_x.shape", t_x.shape)
        return t_x


class decoder_block(nn.Module):
    def __init__(self, debug=False):
        super(decoder_block, self).__init__()
        self.debug = debug
        self.decoder_block = nn.Sequential(
            DeConvBlock3D(nf[4], nf[3], [7, 3, 3], [2, 2, 2], [2, 1, 1]),
            DeConvBlock3D(nf[3], nf[2], [7, 3, 3], [2, 2, 2], [2, 1, 1])
        )

    def forward(self, x):
        if self.debug:
            print("Decoder")
            print("x.shape", x.shape)
        x = self.decoder_block(x)
        if self.debug:
            print("x.shape", x.shape)
        return x



class iBVPNet_NN(nn.Module):
    def __init__(self, frames, in_channels=3, debug=False):
        super(iBVPNet_NN, self).__init__()
        self.debug = debug

        self.in_channels = in_channels
        if self.in_channels == 1 or self.in_channels == 3:
            self.norm = nn.InstanceNorm3d(self.in_channels)
        elif self.in_channels == 4:
            self.rgb_norm = nn.InstanceNorm3d(3)
            self.thermal_norm = nn.InstanceNorm3d(1)
        else:
            print("Unsupported input channels")

        self.ibvpnet = nn.Sequential(
            encoder_block(in_channels, debug),
            decoder_block(debug),
            # spatial adaptive pooling
            nn.AdaptiveMaxPool3d((frames, 1, 1)),
            nn.Conv3d(nf[2], 1, [1, 1, 1], stride=1, padding=0)
        )

        
    def forward(self, x): # [batch, Features=3, Temp=frames, Width=32, Height=32]
        
        [batch, channel, length, width, height] = x.shape

        x = torch.diff(x, dim=2)

        if self.debug:
            print("Input.shape", x.shape)

        if self.in_channels == 1:
            x = self.norm(x[:, -1:, :, :, :])
        elif self.in_channels == 3:
            x = self.norm(x[:, :3, :, :, :])
        elif self.in_channels == 4:
            rgb_x = self.rgb_norm(x[:, :3, :, :, :])
            thermal_x = self.thermal_norm(x[:, -1:, :, :, :])
            x = torch.concat([rgb_x, thermal_x], dim = 1)
        else:
            try:
                print("Specified input channels:", self.in_channels)
                print("Data channels", channel)
                assert self.in_channels <= channel
            except:
                print("Incorrectly preprocessed data provided as input. Number of channels exceed the specified or default channels")
                print("Default or specified channels:", self.in_channels)
                print("Data channels [B, C, N, W, H]", x.shape)
                print("Exiting")
                exit()

        if self.debug:
            print("Diff Normalized shape", x.shape)

        feats = self.ibvpnet(x)
        if self.debug:
            print("feats.shape", feats.shape)
        rPPG = feats.view(-1, length)
        return rPPG