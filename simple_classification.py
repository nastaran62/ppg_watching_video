import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.utils import shuffle

from utils import validate_predictions
from physiological.feature_extraction import get_ppg_features, display_signal


def pca_classification(physiological_data, labels, classes):
    ppg_data = physiological_data[:, :, :]
    print(physiological_data.shape, labels.shape, ppg_data.shape)
    ppg_train, ppg_test, \
        y_train, y_test = \
        normal_train_test_split(ppg_data, labels)

    scaler = StandardScaler()

    # Fit on training set only.
    scaler.fit(ppg_train)  # Apply transform to both the training set and the test set.
    ppg_train = scaler.transform(ppg_train)
    ppg_test = scaler.transform(ppg_test)

    # Make an instance of the Model
    pca = PCA(.95)
    pca.fit(ppg_train)
    ppg_train = pca.transform(ppg_train)
    ppg_test = pca.transform(ppg_test)

    preds_ppg = \
        physiological_classification(ppg_train, ppg_test, y_train, y_test, classes)
    print("ppg ppg ppg")
    ppg_result = validate_predictions(preds_ppg, y_test, classes)
    print(ppg_result)


def feature_classification(physiological_data, labels, part_seconds, classes, sampling_rate=128):

    # physiological_data, labels = \
    #    prepare_experimental_data(classes, label_type, sampling_rate, ignore_time)
    participants, trials = np.array(labels).shape
    participants, trials, points = physiological_data.shape
    all_physiological_features = []
    if part_seconds == 0:
        part_length = points
        part_count = 1
    else:
        part_length = part_seconds * sampling_rate
        part_count = int(points / part_length)
    all_participants_labels = []
    for p in range(participants):
        all_trials_physiological = []
        all_trial_labels = []
        for t in range(trials):
            physiological_parts = []
            all_parts_labels = []
            for i in range(part_count):
                physiological_parts.append(get_ppg_features(
                    physiological_data[p, t, i*part_length:(i+1)*part_length]))
                all_parts_labels.append(labels[p, t])
            all_trial_labels.append(all_parts_labels)
            all_trials_physiological.append(physiological_parts)
        all_participants_labels.append(all_trial_labels)
        all_physiological_features.append(all_trials_physiological)
    physiological_data = np.array(all_physiological_features)
    all_participants_labels = np.array(all_participants_labels)

    #physiological_train, physiological_test, \
    #    y_train, y_test = \
    #    leave_one_subject_out_split(physiological_data, all_participants_labels)

    physiological_train, physiological_test, \
        y_train, y_test = \
        normal_train_test_split(physiological_data, all_participants_labels)

    #physiological_train, physiological_test, \
    #    y_train, y_test = \
    #    leave_one_trial_out_split(physiological_data,
    #                              np.array(all_participants_labels), trial_out=0)

    preds_physiological = \
        physiological_classification(
            physiological_train, physiological_test, y_train, y_test, classes)
    print(validate_predictions(preds_physiological, y_test, classes))


def physiological_classification(x_train, x_test, y_train, y_test, classes):
    train_mean = np.mean(x_train)
    train_std = np.std(x_train)
    x_train = (x_train - train_mean) / train_std
    x_test = (x_test - train_mean) / train_std

    clf = RandomForestClassifier(n_estimators=200, max_features="auto", class_weight='balanced')
    clf.fit(x_train, y_train)
    pred_values = clf.predict_proba(x_test)
    return pred_values


def normal_train_test_split(physiological_data, labels):
    physiological_features = \
        physiological_data.reshape(-1, physiological_data.shape[-1])
    labels = \
        np.array(labels).reshape(-1)
    physiological_features, labels = \
        shuffle(np.array(physiological_features),
                np.array(labels),
                random_state=100)

    # Trial-based splitting
    physiological_train, physiological_test, y_train, y_test = \
        train_test_split(np.array(physiological_features),
                         np.array(labels),
                         test_size=0.3,
                         random_state=100,
                         stratify=labels)
    return physiological_train, physiological_test, \
        y_train, y_test


def leave_one_subject_out_split(physiological_data, labels, subject_out=0):
    participants_count, trials, parts, features = physiological_data.shape

    physiological_train = \
        np.concatenate((physiological_data[0:subject_out, :, :, :],
                        physiological_data[subject_out+1:participants_count, :, :, :]),
                       axis=0)
    print(labels.shape)
    y_train = \
        np.concatenate((labels[0:subject_out, :],
                        labels[subject_out+1:participants_count, :]),
                       axis=0)

    physiological_test = physiological_data[subject_out, :, :, :]
    y_test = labels[subject_out, :]

    physiological_train = \
        physiological_train.reshape(-1, physiological_train.shape[-1])

    physiological_test = \
        physiological_test.reshape(-1, physiological_test.shape[-1])
    y_test = \
        np.array(y_test).reshape(-1)
    y_train = \
        np.array(y_train).reshape(-1)

    physiological_train, y_train = \
        shuffle(physiological_train,
                y_train)

    scaler = MinMaxScaler()
    # Fit on training set only.
    # Apply transform to both the training set and the test set.
    scaler.fit(physiological_train)
    physiological_train = scaler.transform(physiological_train)
    physiological_test = scaler.transform(physiological_test)

    return physiological_train, physiological_test, \
        y_train, y_test


def leave_one_trial_out_split(physiological_data, labels, trial_out=0):
    participants_count, trials, parts, features = physiological_data.shape

    physiological_train = \
        np.concatenate((physiological_data[:, 0:trial_out, :, :],
                        physiological_data[:, trial_out+1:trials, :, :]),
                       axis=1)
    y_train = \
        np.concatenate((labels[:, 0:trial_out],
                        labels[:, trial_out+1:trials]),
                       axis=1)

    physiological_test = physiological_data[:, trial_out, :, :]
    y_test = labels[:, trial_out]

    physiological_train = \
        physiological_train.reshape(-1, physiological_train.shape[-1])

    physiological_test = \
        physiological_test.reshape(-1, physiological_test.shape[-1])
    y_test = \
        np.array(y_test).reshape(-1)
    y_train = \
        np.array(y_train).reshape(-1)

    physiological_train, y_train = \
        shuffle(physiological_train,
                y_train)

    scaler = MinMaxScaler()
    # Fit on training set only.
    # Apply transform to both the training set and the test set.
    scaler.fit(physiological_train)
    physiological_train = scaler.transform(physiological_train)
    physiological_test = scaler.transform(physiological_test)

    return physiological_train, physiological_test, \
        y_train, y_test


def visualize_physiological_data(physiological_features, emotion_labels):
    dict = {"ppg_mean": physiological_features[:, 0],
            "ppg_std": physiological_features[:, 1],
            "ppg_mean": physiological_features[:, 2],
            "ppg_std": physiological_features[:, 3],
            "emotions": emotion_labels, }

    data = pd.DataFrame(dict)
    sns.set_style("whitegrid")
    # sns.FacetGrid(data, hue="emotions", size=4) \
    #   .map(plt.scatter, "ppg_mean", "ppg_std") \
    #   .add_legend()
    sns.pairplot(data, hue="emotions", size=4)

    plt.show()


def kfold_testing(physiological_data, labels, part_seconds, classes, sampling_rate=128):
    participants, trials = np.array(labels).shape
    participants, trials, points = physiological_data.shape
    all_physiological_features = []
    if part_seconds == 0:
        part_length = points
        part_count = 1
    else:
        part_length = part_seconds * sampling_rate
        step = 1 * sampling_rate
        part_count = int((points - part_length) / step) + 1
    all_participants_labels = []
    for p in range(participants):
        all_trials_physiological = []
        all_trial_labels = []
        for t in range(trials):
            physiological_parts = []
            all_parts_labels = []
            start = 0
            end = part_length
            for i in range(part_count):
                physiological_parts.append(get_ppg_features(
                    physiological_data[p, t, start:end]))
                start = start + step
                end = start + part_length
                all_parts_labels.append(labels[p, t])
            all_trial_labels.append(all_parts_labels)
            first_index = 0
            first_true = False
            for i in range(len(physiological_parts)):
                if physiological_parts[i] == []:
                    physiological_parts[i] = physiological_parts[i-1]
                elif first_true is False:
                    first_index = i
                    first_true = True
            i = 0
            while i < first_index:
                physiological_parts[i] = physiological_parts[first_index]
                i += 1
            physiological_parts = np.array(physiological_parts)
            all_trials_physiological.append(physiological_parts)
        all_participants_labels.append(all_trial_labels)
        all_physiological_features.append(all_trials_physiological)
    physiological_data = np.array(all_physiological_features)
    all_participants_labels = np.array(all_participants_labels)

    print(physiological_data.shape, np.array(all_participants_labels).shape)
    physiological_features = \
        physiological_data.reshape(-1, physiological_data.shape[-1])
    labels = \
        np.array(all_participants_labels).reshape(-1)
    # CLASSES COUNT
    CLASSES = [0, 1]
    for i in range(len(CLASSES)):
        print("class count", CLASSES[i], (np.array(labels) == CLASSES[i]).sum())
    k = 5
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=100)
    sum_fscore = 0
    sum_accuracy = 0
    fold = 0
    print(labels.shape)
    print(physiological_features.shape)
    for train_index, test_index in kf.split(labels, y=labels):
        physiological_train, physiological_test = \
            physiological_features[train_index, :], physiological_features[test_index, :]

        y_train, y_test = labels[train_index], labels[test_index]
        print(physiological_train.shape, physiological_test.shape)
        print(y_train.shape, y_test.shape)

        preds_physiological = \
            physiological_classification(
                physiological_train, physiological_test, y_train, y_test, classes)
        accuracy, precision, recall, f_score = \
            validate_predictions(preds_physiological, y_test, classes)
        print("fold, accuracy, precision, recall, f_score",
              fold, accuracy, precision, recall, f_score)
        sum_fscore += f_score
        sum_accuracy += accuracy
        fold += 1
    print("avg_fscore: ", sum_fscore/k, "avg_accuracy: ", sum_accuracy/k)


def kfold_testing_new(physiological_data, labels, part_seconds, classes, sampling_rate=128):
    participants, trials = np.array(labels).shape
    participants, trials, points = physiological_data.shape
    all_physiological_features = []
    if part_seconds == 0:
        part_length = points
        part_count = 1
    else:
        part_length = part_seconds * sampling_rate
        step = 1 * sampling_rate
        part_count = int((points - part_length) / step) + 1
    all_participants_labels = []
    for p in range(participants):
        all_trials_physiological = []
        all_trial_labels = []
        for t in range(trials):
            physiological_parts = []
            all_parts_labels = []
            start = 0
            end = part_length
            for i in range(part_count):
                physiological_parts.append(get_ppg_features(
                    physiological_data[p, t, start:end]))
                start = start + step
                end = start + part_length
                all_parts_labels.append(labels[p, t])
            all_trial_labels.append(all_parts_labels)
            first_index = 0
            first_true = False
            for i in range(len(physiological_parts)):
                if physiological_parts[i] == []:
                    physiological_parts[i] = physiological_parts[i-1]
                elif first_true is False:
                    first_index = i
                    first_true = True
            i = 0
            while i < first_index:
                physiological_parts[i] = physiological_parts[first_index]
                i += 1
            physiological_parts = np.array(physiological_parts)
            all_trials_physiological.append(physiological_parts)
        all_participants_labels.append(all_trial_labels)
        all_physiological_features.append(all_trials_physiological)
    physiological_data = np.array(all_physiological_features)
    all_participants_labels = np.array(all_participants_labels)

    print(physiological_data.shape, np.array(all_participants_labels).shape)
    physiological_features = \
        physiological_data.reshape(-1, *physiological_data.shape[-2:])
    labels = np.array(all_participants_labels).reshape(-1,
                                                       np.array(all_participants_labels).shape[-1])
    print(physiological_features.shape, labels.shape)
    # CLASSES COUNT
    CLASSES = [0, 1]
    for i in range(len(CLASSES)):
        print("class count", CLASSES[i], (np.array(labels) == CLASSES[i]).sum())
    k = 5
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=50)
    sum_fscore = 0
    sum_accuracy = 0
    fold = 0
    print(labels.shape)
    print(physiological_features.shape)
    for train_index, test_index in kf.split(labels, y=labels[:, 0]):
        physiological_train, physiological_test = \
            physiological_features[train_index, :, :], physiological_features[test_index, :, :]

        y_train, y_test = labels[train_index, :], labels[test_index, :]
        print(physiological_train.shape, physiological_test.shape)
        print(y_train.shape, y_test.shape)
        physiological_train = \
            physiological_train.reshape(-1, physiological_train.shape[-1])
        y_train = y_train.reshape(-1)
        physiological_test = \
            physiological_test.reshape(-1, physiological_test.shape[-1])
        y_test = y_test.reshape(-1)
        print(physiological_train.shape, physiological_test.shape)
        print(y_train.shape, y_test.shape)
        preds_physiological = \
            physiological_classification(
                physiological_train, physiological_test, y_train, y_test, classes)
        accuracy, precision, recall, f_score = \
            validate_predictions(preds_physiological, y_test, classes)
        print("fold, accuracy, precision, recall, f_score",
              fold, accuracy, precision, recall, f_score)
        sum_fscore += f_score
        sum_accuracy += accuracy
        fold += 1
    print("avg_fscore: ", sum_fscore/k, "avg_accuracy: ", sum_accuracy/k)
