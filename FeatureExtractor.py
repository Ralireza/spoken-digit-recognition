import cv2
from skimage.feature import hog
from sklearn.decomposition import PCA
from python_speech_features import mfcc, logfbank, delta
from scipy.io import wavfile


class FeatureExtractor:
    def __init__(self):

        print("\n----------------------------------------------------------")
        print("--------------P-R-O-C-E-S-S-I-N-G---D-A-T-A---------------")
        print("----------------------------------------------------------\n")

    def choose(self, function, sound_path):
        if function == "mfcc":
            return self.mfcc(sound_path)
        if function == "logfbank":
            return self.logfbank(sound_path)
        if function == "delta":
            return self.delta(sound_path)

    def mfcc(self, sound_path):
        sampling_freq, audio = wavfile.read(sound_path)
        mfcc_features = mfcc(audio, sampling_freq)

        return mfcc_features

    def logfbank(self, sound_path):
        sampling_freq, audio = wavfile.read(sound_path)
        filterbank_features = logfbank(audio, sampling_freq)
        return filterbank_features

    def delta(self, sound_path):
        sampling_freq, audio = wavfile.read(sound_path)
        mfcc_features = mfcc(audio, sampling_freq)
        d_mfcc_feat = delta(mfcc_features, 10)
        return d_mfcc_feat
