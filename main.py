import itertools
import os
from FeatureExtractor import FeatureExtractor
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from hmmlearn import hmm
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank, delta
import matplotlib.pyplot as plt


def build_dataset(sound_path='spoken_digit/', fe="delta"):
    feature_class = FeatureExtractor()
    files = sorted(os.listdir(sound_path))
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    data = dict()
    i = 0

    for f in files:
        feature = feature_extractor(sound_path=sound_path + f)
        if i % 5 == 0:
            x_test.append(feature)
            y_test.append(int(f[0]))
        else:
            x_train.append(feature)
            y_train.append(f[0])
        i += 1

    for i in range(0, len(x_train), len(x_train) // 10):
        data[y_train[i]] = x_train[i:i + len(x_train) // 10]
    return x_train, y_train, x_test, y_test, data


def feature_extractor(sound_path):
    sampling_freq, audio = wavfile.read(sound_path)
    mfcc_features = mfcc(audio, sampling_freq)

    return mfcc_features

def train_model(data):
    learned_hmm = dict()
    for label in data.keys():
        model = hmm.GMMHMM(n_components=5)
        feature = np.ndarray(shape=(1, 13))
        for list_feature in data[label]:
            feature = np.vstack((feature, list_feature))
        obj = model.fit(feature)
        learned_hmm[label] = obj
    return learned_hmm


def prediction(test_data):
    predict_label = []
    for test in test_data:
        scores = []
        for node in learned_hmm.keys():
            scores.append(learned_hmm[node].score(test))
        predict_label.append(scores.index(max(scores)))
    return predict_label

