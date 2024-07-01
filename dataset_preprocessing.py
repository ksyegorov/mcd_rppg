import sys, os
from glob import glob
import numpy as np
import pandas as pd


import rppglib.data_utils
import rppglib.face_utils


VIDEOS_PATH = '/gim/lv03/datasets/rppg/rppg_nir_data_train/video/'
FACES_PATH = '/gim/lv03/datasets/rppg/rppg_nir_data_train/faces/'
LANDMARKS_PATH = '/gim/lv03/datasets/rppg/rppg_nir_data_train/landmarks/'

errors_file = 'errors.csv'

if not os.path.isfile(errors_file):
    errors = pd.DataFrame({'video_file': [], 'error_type': [], 'error_msg': []})
    errors.to_csv(errors_file, index=False)

videos = sorted(glob(f'{VIDEOS_PATH}*.avi'))  
print(f'Total video count: {len(videos)}')

errors = pd.read_csv(errors_file)

for video_file in videos:
    print(f'Processing {video_file}')
    
    face_file = video_file.replace(VIDEOS_PATH, FACES_PATH) + '.npy'
    landmarks_file = video_file.replace(VIDEOS_PATH, LANDMARKS_PATH) + '.npy'

    if os.path.isfile(face_file) and os.path.isfile(landmarks_file):
        continue
    if video_file in errors['video_file'].values:
        continue

    
    try:
        video  = rppglib.data_utils.load_video(video_file)
        video, landmarks = rppglib.face_utils.process_video(video)
        np.save(face_file, video)
        np.save(landmarks_file, landmarks)
        
    except AssertionError as e:
        error_row = {'video_file': video_file, 'error_type': str(type(e)), 'error_msg': str(e)}
        errors = pd.concat([errors, pd.DataFrame([error_row])], ignore_index=True)
        errors.to_csv(errors_file, index=False)
        print(e)