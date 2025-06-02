import sys, os
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.io
import matplotlib.pyplot as plt
import cv2
import torch
from sklearn.model_selection import KFold, train_test_split

import rppglib.data_utils
import rppglib.face_utils
import rppglib.params
import rppglib.processing
import rppglib.models
import rppglib.ppg2hr

class rPPG_Dataset(torch.utils.data.Dataset):
    def __init__(self, preprocessed_files, video_processing, config):
        self.config = config
        self.preprocessed_files = preprocessed_files
        self.video_processing = video_processing

    def __len__(self, ):
        return len(self.preprocessed_files)

    def __getitem__(self, index):
        npz = np.load(self.preprocessed_files[index])
        video = npz['video']
        landmarks = npz['landmarks']
        ppg = npz['ppg']

        video = self.preprocess_video(video, landmarks)
        ppg = self.preprocess_ppg(ppg)
        
        return video, ppg

    def preprocess_video(self, video, landmarks):
        video = self.video_processing(video, landmarks)
        return video

    def preprocess_ppg(self, ppg):
        ppg = rppglib.processing.bandpass_filter(ppg, self.config.fps, self.config.ppg_low_freq, self.config.ppg_high_freq)
        ppg -= ppg.mean()
        ppg /= ppg.std() + 1e-9    
        return ppg


def train_fold(config):
    model = rppglib.models.__dict__[config.model](config)
    print(f'Model {config.model} loaded')
    
    if model.require_train:
        
        print(f'Training on fold {config.valid_fold}')
        df = pd.read_csv(f'{config.train_dataset}.csv')

        test_mask = df['fold'] == config.test_fold
        valid_mask = df['fold'] == config.valid_fold
        train_mask = (~test_mask)&(~valid_mask)
    
        print('Train files:', train_mask.sum())
        print('Valid files:', valid_mask.sum())
        
        train_ds = rPPG_Dataset(df.loc[train_mask, 'file'].values, model.video_processing, config)
        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
        
        valid_ds = rPPG_Dataset(df.loc[valid_mask, 'file'].values, model.video_processing, config)
        valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
        model.train(train_dl, valid_dl)

    config.test_results = dict()
    for test_dataset in config.test_datasets:
        print()
        print(f'Testing on {test_dataset}')
        df = pd.read_csv(f'{test_dataset}.csv')
        test_mask = df['fold'] == config.test_fold
        print('Test files:', test_mask.sum())
        test_ds = rPPG_Dataset(df.loc[test_mask, 'file'].values, model.video_processing, config)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)


        true_ppgs_epoch = list()
        pred_ppgs_epoch = list()

        for videos, true_ppgs in tqdm(test_dl, desc='Testing', file=sys.stdout):
            assert not torch.isnan(videos).any(), 'NaNs in Videos'
            assert not torch.isnan(true_ppgs).any(), 'NaNs in PPGs'
            pred_ppgs = model.predict(videos)
            assert not torch.isnan(pred_ppgs).any(), 'NaNs in pred PPGs'
            
            #ppg_mae, hr_mae = clac_metrics(true_ppgs, pred_ppgs)
            #ppg_maes += ppg_mae
            #hr_maes += hr_mae
            
            true_ppgs_epoch.append(true_ppgs.detach().cpu().numpy())
            pred_ppgs_epoch.append(pred_ppgs.detach().cpu().numpy())

        
        true_ppgs_epoch = np.concatenate(true_ppgs_epoch, axis=0)
        pred_ppgs_epoch = np.concatenate(pred_ppgs_epoch, axis=0)


        true_ppgs_epoch = true_ppgs_epoch[:, :pred_ppgs_epoch.shape[1]]
        print('Testing true PPGs:', true_ppgs_epoch.shape)
        print('Testing pred PPGs:', pred_ppgs_epoch.shape)

        ppg_mae, hr_mae = calc_metrics(true_ppgs_epoch, pred_ppgs_epoch)
        
        config.test_results[f'test__{test_dataset}__ppg'] = ppg_mae
        config.test_results[f'test__{test_dataset}__hr'] = hr_mae
        print('PPG MAE:', ppg_mae)
        print('HR MAE:', hr_mae)
    return config

def calc_metrics(true_ppgs, pred_ppgs):

    #np.save('true_ppgs.npy', true_ppgs)
    #np.save('pred_ppgs.npy', pred_ppgs)
    #assert False

    # plt.figure(figsize=(20, 3))
    # plt.plot(true_ppgs[0])
    # plt.plot(pred_ppgs[0])
    # plt.show()

    # plt.figure(figsize=(20, 3))
    # plt.plot(true_ppgs[-1])
    # plt.plot(pred_ppgs[-1])
    # plt.show()
 
    ppg_mae = np.abs(pred_ppgs - true_ppgs).mean(axis=1).mean().item()

    true_hrs = list()
    pred_hrs = list()
    for i in range(true_ppgs.shape[0]):
        true = true_ppgs[i]
        pred = pred_ppgs[i]

        true_hr = rppglib.processing.calculate_fft_hr(true)
        pred_hr = rppglib.processing.calculate_fft_hr(pred)

        true_hrs.append(true_hr)
        pred_hrs.append(pred_hr)
    
        # true_hr, _ = rppglib.ppg2hr.BVPsignal(true, 30).getBPM(winsize=20)
        # pred_hr, _ = rppglib.ppg2hr.BVPsignal(pred, 30).getBPM(winsize=20)
        # true_hr = true_hr.mean()
        # pred_hr = pred_hr.mean()

    print('Mean true HRs:', np.mean(true_hrs))
    print('Mean pred HRs:', np.mean(pred_hrs))
        
        
    hr_mae = np.abs(np.array(true_hrs) - np.array(pred_hrs)).mean()
    return ppg_mae, hr_mae