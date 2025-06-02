import numpy as np
from scipy.signal import  stft

class BVPsignal:
    """
    Manage (multi-channel, row-wise) BVP signals, and transforms them in BPMs.
    """
    #nFFT = 2048  # freq. resolution for STFTs
    step = 1       # step in seconds

    def __init__(self, data, fs, startTime=0, minHz=0.75, maxHz=4., verb=False):
        if len(data.shape) == 1:
            self.data = data.reshape(1, -1)  # 2D array raw-wise
        else:
            self.data = data
        self.fs = fs                       # sample rate
        self.startTime = startTime
        self.verb = verb
        self.minHz = minHz
        self.maxHz = maxHz
        nyquistF = self.fs/2
        fRes = 0.5
        self.nFFT = max(2048, (60*2*nyquistF) / fRes)

    def spectrogram(self, winsize=5):
        """
        Compute the BVP signal spectrogram restricted to the
        band 42-240 BPM by using winsize (in sec) samples.
        """

        # -- spect. Z is 3-dim: Z[#chnls, #freqs, #times]
        F, T, Z = stft(self.data,
                       self.fs,
                       nperseg=self.fs*winsize,
                       noverlap=self.fs*(winsize-self.step),
                       boundary='even',
                       nfft=self.nFFT)
        Z = np.squeeze(Z, axis=0)

        # -- freq subband (0.65 Hz - 4.0 Hz)
        minHz = 0.65
        maxHz = 4.0
        band = np.argwhere((F > minHz) & (F < maxHz)).flatten()
        self.spect = np.abs(Z[band, :])     # spectrum magnitude
        self.freqs = 60*F[band]            # spectrum freq in bpm
        self.times = T                     # spectrum times

        # -- BPM estimate by spectrum
        self.bpm = self.freqs[np.argmax(self.spect, axis=0)]

    def getBPM(self, winsize=5):
        """
        Get the BPM signal extracted from the ground truth BVP signal.
        """
        self.spectrogram(winsize)
        return self.bpm, self.times