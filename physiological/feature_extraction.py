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
