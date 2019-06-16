<p align="center"><img width=40% src="https://github.com/ralireza/spoken-digit-recognition/blob/master/logo.png"></p>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
![Python](https://img.shields.io/badge/python-v3.6+-blue.svg)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg)
[![License](https://img.shields.io/github/license/ralireza/PHDR.svg)](https://opensource.org/licenses/MIT)



# spoken-digit-recognition
Classifying English spoken digit by Hidden Markov Model 

# Classifier 
[HMM](https://en.wikipedia.org/wiki/Hidden_Markov_model) - Hidden Markov Model


# Feature Extractor
[MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum) - Mel-frequency Cepstrum

# Accuracy
0.98%

## General Step
this is for curious guys and implement by themselves.
* Downlaod dataset from [Kaggle](https://www.kaggle.com/divyanshu99/spoken-digit-dataset)
* Extract feature of each data with mfcc
* Train Hmm states by [hmmlearn](https://hmmlearn.readthedocs.io/en/latest/)
* Predict test data
* Evaluate the model

# Hands on code
this is for lazy progrmmer and easy understand whole of project at one look.
* ## Parsing data and extract feature from them in this way (0.20 % for test)
```python
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

x_train, y_train, x_test, y_test, data = build_dataset()
```
* ## we give data to train hmm
```python
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
    
learned_hmm = train_model(data)

```

* ## Save learned hmm to pickle and Speed up the test phase (after first run comment this lines) :
```python
with open("learned.pkl", "wb") as file:
     pickle.dump(learned_hmm, file)
```
* ## clever guy can guess this step ;) -> read from pickle:
 ```python
 with open("learned.pkl", "rb") as file:
    learned_hmm = pickle.load(file)
```

* ## prediction:
```python
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
    

y_pred = prediction(x_test, learned_hmm)

```

* ## Best part is evaluate our model:
```python
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

report(y_test, y_pred, show_cm=True)
```
if show_cm is True , i'll  magically plot a cool confusion matrix for U <3.
<p align="center"><img width=40% src="https://github.com/ralireza/spoken-digit-recognition/blob/master/cm.png"></p>
#### Special thanks to [@samadvalipour](https://github.com/samadvalipour)
