from neurokit2.hrv.hrv_nonlinear import hrv_nonlinear
import pywt
import numpy as np
import matplotlib.pyplot as plt
from pywt import wavedec
import heartpy as hp
PPG_SAMPLING_RATE =128

import neurokit2 as nk 

def get_ppg_features(ppg_data):
    # wd, dict_data = hp.process(ppg_data, 128)
    # feature =  list(dict_data.values())
    # print(feature[1:6])   

##################################################################################
# Static Values
#################################################################################
    hrv_time = nk.hrv_time(ppg_data, sampling_rate=128, show=True)

    HRV_MadNN = hrv_time['HRV_MadNN'].values.tolist()

    HRV_MCVNN = hrv_time['HRV_MCVNN'].values.tolist()

    HRV_IQRNN = hrv_time['HRV_IQRNN'].values.tolist()

    HRV_MeanNN = [np.mean(ppg_data)]

    HRV_SDNN = [np.std(ppg_data)]


    temp = HRV_IQRNN + HRV_MadNN + HRV_MCVNN + HRV_MeanNN + HRV_SDNN 

    print(temp)
    
    return np.array(temp)

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
