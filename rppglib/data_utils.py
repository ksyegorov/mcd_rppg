import skvideo.io
import numpy as np

import numpy
numpy.float = numpy.float64
numpy.int = numpy.int_

def load_video(path):
    videodata = skvideo.io.vread(path) 
    return videodata


def load_ppg(path):
    with open(path, 'rt') as f:
        ppg = f.read().split('\n')[:-1]
    ppg = np.array([int(line.split()[0]) for line in ppg]).astype('float32')
    return ppg