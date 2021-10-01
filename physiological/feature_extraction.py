import neurokit2 as nk
from neurokit2.hrv.hrv_nonlinear import hrv_nonlinear
import pywt
import numpy as np
import matplotlib.pyplot as plt
from pywt import wavedec
import heartpy as hp
PPG_SAMPLING_RATE = 128


def display_signal(signal):
    plt.plot(signal)
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()


def get_ppg_features(ppg_data):
    try:
        wd, dict_data = hp.process(ppg_data, 128)
        # print(wd)
        # print("**************************")
        # print(dict_data)
        # print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        # feature = list(dict_data.values())
        # print(feature[1:6])

        ##################################################################################
        # Static Values
        #################################################################################
        # peaks = nk.signal_rate(wd["peaklist"], sampling_rate=128,
        #                       desired_length=len(ppg_data))

        # peaks = info["PPG_Peaks"]

        # hrv_time = nk.hrv_time(peaks, sampling_rate=128, show=True)
        # print(hrv_time)

        # hrv_freq = nk.hrv_frequency(peaks, sampling_rate=128, show=False)

        # print(hrv_freq)
        '''
        hrv_madnn = hrv_time['HRV_MadNN'].values.tolist()

        hrv_mcvnn = hrv_time['HRV_MCVNN'].values.tolist()

        hrv_iqrnn = hrv_time['HRV_IQRNN'].values.tolist()
        '''
        ppg_mean = [np.mean(ppg_data)]

        ppg_std = [np.std(ppg_data)]

        temp = [np.mean(ppg_data), np.std(ppg_data),
                dict_data["bpm"], dict_data["ibi"], dict_data["sdnn"],
                dict_data["sdsd"], dict_data["bpm"]]
        if np.nan in temp or np.ma.masked in temp:
            return []
        else:
            temp = np.array(temp)
    except Exception as error:
        temp = []

    return temp


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
