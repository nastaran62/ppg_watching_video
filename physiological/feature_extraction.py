import pywt
import numpy as np
import matplotlib.pyplot as plt
from pywt import wavedec


def get_ppg_features(ppg_data):
    ppg_features = [np.mean(ppg_data), np.std(ppg_data)]

    feature = ppg_features

    return np.array(feature)


def prepare_ppg_components(ppg_data: np.array, sampling_rate: int,
                           window_size: int = 20, overlap: int = 19):
    '''
    Extracts HR, HRV and breathing rate from PPG

    @param np.array ppg_data: PPG data
    @param int sampling_rate: PPG sampling rate

    @keyword int window_length: Length of sliding window for measurment in seconds
    @keyword float overlap: Amount of overlap between two windows in seconds

    @rtype: dict(str: numpy.array)
    @note: dict.keys = ["hr", "hrv", "breathing_rate"]

    @return a dictionary of PPG components
    '''

    wd, m = hp.process_segmentwise(data,
                                   sample_rate=sampling_rate,
                                   segment_width=window_size,
                                   segment_overlap=overlap/window_size)

    hr_components = {"hr": np.array(m['bpm']),
                     "hrv": np.array(m['sdsd']),
                     "breathing_rate": np.array(m['breathingrate'])}

    return hr_components


def _get_multimodal_statistics(signal_data):
    mean = np.mean(signal_data, axis=1)
    std = np.std(signal_data, axis=1)
    return [mean, std]
