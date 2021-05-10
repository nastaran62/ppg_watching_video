import pywt
import numpy as np
import matplotlib.pyplot as plt
from pywt import wavedec


def get_ppg_features(ppg_data):
    ppg_features = [np.mean(ppg_data), np.std(ppg_data), np.var(ppg_data), np.max(ppg_data), np.min(ppg_data)]

    feature = ppg_features

    return np.array(feature)


def _get_frequency_features(data):

    (cA, cD) = pywt.dwt([1, 2, 3, 4, 5, 6], 'db1')

    bands = [cA, cD]
    all_features = []
    for band in range(len(bands)):
        power = np.sum(bands[band]**2)
        entropy = np.sum((bands[band]**2)*np.log(bands[band]**2))
        all_features.extend([power, entropy])
    return all_features


def _get_multimodal_statistics(signal_data):
    mean = np.mean(signal_data, axis=1)
    std = np.std(signal_data, axis=1)
    return [mean, std]
