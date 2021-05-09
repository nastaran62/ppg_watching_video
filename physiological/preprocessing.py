import numpy as np
import heartpy as hp
import matplotlib.pyplot as plt
import scipy


def display_signal(signal):
    plt.plot(signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def display_signal2(signal):
    fig1, ax1 = plt.subplots(2, sharex=False, sharey=False)
    ax1[0].plot(signal)
    ax1[0].set(ylabel="amplitude")

    ax1[1].plot(signal)
    ax1[1].set(ylabel="ppg")
    maxima, = scipy.signal.argrelextrema(np.array(signal), np.greater)
    minima, = scipy.signal.argrelextrema(np.array(signal), np.less)
    ax1[1].scatter(maxima, np.array(signal)[maxima], linewidth=0.03, s=50, c='r')
    ax1[1].scatter(minima, np.array(signal)[minima], linewidth=0.03, s=50, c='g')

    plt.show()


def physiological_preprocessing(physiological_data, sampling_rate=128):
    '''
    Preprocesss ppg
    '''

    preprocessed_ppg = ppg_preprocessing(physiological_data,
                                         sampling_rate)
    data = normalization(np.array(preprocessed_ppg))
    # display_signal2(normalization(np.array(preprocessed_ppg)))

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
