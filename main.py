import itertools
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from hmmlearn import hmm
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank, delta
import matplotlib.pyplot as plt
import pickle
import speech_recognition as sr


def build_dataset(sound_path='spoken_digit/'):
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
        model = hmm.GMMHMM(n_components=14)
        feature = np.ndarray(shape=(1, 13))
        for list_feature in data[label]:
            feature = np.vstack((feature, list_feature))
        obj = model.fit(feature)
        learned_hmm[label] = obj
    return learned_hmm


def prediction(test_data, trained):
    # predict list of test
    predict_label = []
    if type(test_data) == type([]):
        for test in test_data:
            scores = []
            for node in trained.keys():
                scores.append(trained[node].score(test))
            predict_label.append(scores.index(max(scores)))
    # predict a test
    else:
        scores = []
        for node in trained.keys():
            scores.append(trained[node].score(test_data))
        predict_label.append(scores.index(max(scores)))
    return predict_label


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def report(y_test, y_pred, show_cm=True):
    print("confusion_matrix:\n\n", confusion_matrix(y_test, y_pred))
    print("----------------------------------------------------------")
    print("----------------------------------------------------------\n")
    print("classification_report:\n\n", classification_report(y_test, y_pred))
    print("----------------------------------------------------------")
    print("----------------------------------------------------------\n")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("----------------------------------------------------------")
    print("----------------------------------------------------------\n")
    if show_cm:
        plot_confusion_matrix(confusion_matrix(y_test, y_pred), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


x_train, y_train, x_test, y_test, data = build_dataset()
#
# learned_hmm = train_model(data)
# with open("learned.pkl", "wb") as file:
#     pickle.dump(learned_hmm, file)

with open("learned.pkl", "rb") as file:
    learned_hmm = pickle.load(file)
single_test = feature_extractor('./1-1.wav')


# y_pred = prediction(x_test, learned_hmm)
# report(y_test, y_pred, show_cm=True)
y_pred = prediction(single_test, learned_hmm)
print(y_pred)
