import numpy as np
import heartpy as hp
import matplotlib.pyplot as plt


def display_signal(signal):
    plt.plot(signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def physiological_preprocessing(physiological_data, sampling_rate=128):
    '''
    Preprocesss ppg
    '''

    preprocessed_ppg = ppg_preprocessing(physiological_data,
                                         sampling_rate)
    data = normalization(np.array(preprocessed_ppg))
    # display_signal(normalization(np.array(preprocessed_ppg)))

    return data


def ppg_preprocessing(data, sampling_rate, low_pass=0.7, high_pass=2.5):
    filtered = hp.filter_signal(data,
                                [low_pass, high_pass],
                                sample_rate=sampling_rate,
                                order=3,
                                filtertype='bandpass')

    return filtered


def normalization(data):
    # Normalization
    min = np.amin(data)
    max = np.amax(data)
    output = (data - min) / (max - min)
    return output
