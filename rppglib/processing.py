import scipy.signal
import numpy as np
import cv2

from . import face_utils

def filter_signal(signal, rate, freq, mode='high', order=4):
    hb_n_freq = freq / (rate / 2)
    b, a = scipy.signal.butter(order, hb_n_freq, mode)
    filtered = scipy.signal.filtfilt(b, a, signal)
    filtered = filtered.astype(signal.dtype)
    return filtered

def bandpass_filter(signal, rate, low_freq, high_freq, order=4):
    signal = filter_signal(signal, rate, high_freq, mode='low',  order=order)
    signal = filter_signal(signal, rate, low_freq,  mode='high', order=order)
    return signal
    

def prepare_data(video, ppg, sample_rate, window, ppg_low_freq, ppg_high_freq, max_chunks, cache_folder, dataset_name, subject_name, video_name):
    assert video.shape[0] == ppg.shape[0], f'Video shape: {video.shape}, PPG shape: {ppg.shape}'
    assert len(video.shape) == 4, f'Video shape: {video.shape}'
    assert len(ppg.shape) == 1, f'PPG shape: {ppg.shape}'
    assert video.shape[3] == 3, f'Video shape: {video.shape}'
    assert video.dtype == np.uint8, f'Video dtype: {video.dtype}'
    assert ppg.dtype == np.float32, f'PPG dtype: {ppg.dtype}'

    data = list()
    num_chunks = min(video.shape[0] // window, max_chunks)
    for chunk_id in range(num_chunks):
        start = window * chunk_id
        end = start + window
        video_chunk = video[start: end].copy()
        video_chunk = face_utils.crop_face_video(video_chunk)
        ppg_chunk = ppg[start: end].copy()

        ppg_chunk = bandpass_filter(ppg_chunk, sample_rate, ppg_low_freq, ppg_high_freq, order=4)
        ppg_chunk -= ppg_chunk.mean()
        ppg_chunk /= ppg_chunk.std() + 1e-9

        hr = int(np.round(calculate_fft_hr(ppg_chunk, fs=sample_rate)))

        video_chunk_filename = f'{cache_folder}/{dataset_name}_{subject_name}_{video_name}_{chunk_id}_HR_{hr}_vid.npy'
        ppg_chunk_filename   = f'{cache_folder}/{dataset_name}_{subject_name}_{video_name}_{chunk_id}_HR_{hr}_ppg.npy'

        np.save(video_chunk_filename, video_chunk.astype('uint8'))
        np.save(ppg_chunk_filename, ppg_chunk.astype('float32'))



def _next_power_of_2(x):
    """Calculate the nearest power of 2."""
    return 1 if x == 0 else 2 ** (x - 1).bit_length()

def calculate_fft_hr(ppg_signal, fs=30, low_pass=0.50, high_pass=3.0):
    """
    Taken from https://github.com/ubicomplab/rPPG-Toolbox/blob/main/evaluation/post_process.py
    Calculate heart rate based on PPG using Fast Fourier transform (FFT).
    """
    ppg_signal = np.expand_dims(ppg_signal, 0)
    N = _next_power_of_2(ppg_signal.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(ppg_signal, fs=fs, nfft=N, detrend=False)
    fmask_ppg = np.argwhere((f_ppg >= low_pass) & (f_ppg <= high_pass))
    mask_ppg = np.take(f_ppg, fmask_ppg)
    mask_pxx = np.take(pxx_ppg, fmask_ppg)
    fft_hr = np.take(mask_ppg, np.argmax(mask_pxx, 0))[0] * 60
    return fft_hr


def resize_video(video, height, width):
    resized_video = np.zeros((video.shape[0], height, width, 3), dtype=video.dtype)
    for i in range(video.shape[0]):
        resized_video[i] = cv2.resize(video[i], (width, height), interpolation=cv2.INTER_AREA)
    return resized_video

def video_to_rgb(video):
    mask = video != 0
    rgb = np.zeros((video.shape[0], 3), dtype='float32')
    for i in range(video.shape[0]):
        rgb[i, 0] = video[i, :, :, 0][mask[i, :, :, 0]].mean()
        rgb[i, 1] = video[i, :, :, 1][mask[i, :, :, 1]].mean()
        rgb[i, 2] = video[i, :, :, 2][mask[i, :, :, 2]].mean()
    return rgb