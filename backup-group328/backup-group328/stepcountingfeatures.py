import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz, iirnotch, filtfilt, firwin
import sys
import math

def filtSignal(x_signal, y_signal, z_signal, timestamps):
    sampling_rate = 100
    # nyq = sampling_rate/2
    # cutoff = 3.05/nyq
    buffer_window_size = 100

    step_locations = []
    step_baseline = []
    x = []
    y = []
    z = []
    accel_mag = []
    accel_time = []
    steps = 0

    for k in range(0, len(timestamps)):
        x.append(x_signal[k])
        y.append(y_signal[k])
        z.append(z_signal[k])
        accel_time.append(timestamps[k])
        accel_mag.append(math.sqrt(x[k]*2+y[k]*2+z[k]*2))

    for i in range(0, len(accel_mag), buffer_window_size - 1):
        buffer = accel_mag[i:i+buffer_window_size]
        filt = buffer
        if len(buffer)<18:
            break
        else:
            # remove baseline wander caused by breathing
            fs = sampling_rate  # Sample frequency (Hz)
            order = int(0.3 * fs)
            if order % 2 == 0: order += 1

            # the cutoff frequencies for the filter (Hz)
            f1, f2 = .5, .75
            
            # remember to normalize the frequencies to nyquist.
            f1 = 2. * f1 / sampling_rate
            f2 = 2. * f2 / sampling_rate

            a = np.array([1])
            b = firwin(order,[f1, f2],pass_zero=False)
            # not needed? normal_cutoff = cutoff / nyq
            b, a = butter(3, [f1, f2], btype='band')
            # applying the filter to the buffer array
            filt = filtfilt(b, a, buffer)

            max = filt.max()
            min = filt.min()
            dynamic_threshold = (max+min)/2

        for j in range(0, len(filt)-1):
            if((filt[j+1] < dynamic_threshold) and (filt[j-1] > dynamic_threshold) and accel_mag[j] > dynamic_threshold and (accel_mag[j-1] not in step_locations)):
                steps += 1
                step_locations.append(accel_mag[j])
                step_baseline.append(accel_time[j])

    return step_baseline, step_locations, filt, steps


def filter_signal(r):
    fs = 100 #sample rate
    nyq = fs/2
    cutoff = 1.25/nyq
    order = 4
    b, a = butter(order, cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, r)
    windowSize = 10
    meansig = []
    for x in range(0, len(filtered_signal)):
        avgarr = filtered_signal[x:x+windowSize]
        c = np.average(avgarr)
        meansig.append(c)

    peak_locations = []
    peak_baseline =[]
    mean = meansig[0]
    peak = 0
    for j in range(0, len(filtered_signal)-1):
        if(j%windowSize):
            mean=meansig[j]
        if filtered_signal[j] < mean and filtered_signal[j+1]>=mean and not j in peak_locations and not j in peak_locations:
            peak_baseline.append(ts[j])
            peak_locations.append(r[j])
    return peak_baseline, peak_locations, filtered_signal

def getSteps(x_signal, y_signal, z_signal, timestamps):
    #peak_baseline = filter_signal(r)[0]
    steps = filtSignal(x_signal, y_signal, z_signal, timestamps)[3]
    return steps #len(peak_baseline)

def getEntropy(originalSig):
    hist_distribution = np.histogram(originalSig)
    modified_hist_distribution = []
    for h in hist_distribution:
        if(h == 0):
            modified_hist_distribution.append(1)
        else:
            modified_hist_distribution.append(h)
    total_count_acceleration_val = sum(hist_distribution*np.log10(modified_hist_distribution)) #Entropy: sum p(x)log(p(x)), I added 
    #"*np.log10(hind_distribution))" 
    return [total_count_acceleration_val,total_count_acceleration_val,total_count_acceleration_val]

def compute_variance_features(r):
    """
    Computes the variance of x, y, z acceleration over the given window.
    """
    #we can try using a filtered signal first to see if it will get better results:
    # filt_sig = filter_signal(r)[2]
    filt_sig = r
    return [np.var(filt_sig), np.var(filt_sig), np.var(filt_sig)]

def compute_maximum_features(r):
    """
    Computes the max value of x, y, z acceleration over the given window.
    """
    #we can try using a filtered signal first to see if it will get better results:
    # filt_sig = filter_signal(r)[2]
    filt_sig = r
    return [np.amax(filt_sig), np.amax(filt_sig), np.amax(filt_sig)]

def compute_minimum_features(r):
    """
    Computes the min value of x, y, z acceleration over the given window.
    """
    #we can try using a filtered signal first to see if it will get better results:
    # filt_sig = filter_signal(r)[2]
    filt_sig = r
    return [np.amin(filt_sig), np.amin(filt_sig), np.amin(filt_sig)]

def compute_mean_features(r):
    """
    Computes the mean x, y and z acceleration over the given window. 
    """
    return np.mean(r, axis=0)

def extract_features(window):
    """
    Here is where you will extract your features from the data over 
    the given window. We have given you an example of computing 
    the mean and appending it to the feature vector.

    """

    x = []
    feature_names = []
    
    #mean features
    x.append(compute_mean_features(window))
    feature_names.append("x_mean")
    feature_names.append("y_mean")
    feature_names.append("z_mean")

    
    # variance features

    x.append(compute_variance_features(window))
    feature_names.append("x_variance")
    feature_names.append("y_variance")
    feature_names.append("z_variance")
    


    # maximum features
    x.append(compute_maximum_features(window))
    feature_names.append("x_max")
    feature_names.append("y_max")
    feature_names.append("z_max")

 
    # minimum features
    x.append(compute_minimum_features(window))
    feature_names.append("x_min")
    feature_names.append("y_min")
    feature_names.append("z_min")


    # TODO: call functions to compute other features. Append the features to x and the names of these features to feature_names

    #x.append(_compute_fft_features(window))
    #feature_names.append("x_dominate_freq")
    #feature_names.append("y_dominant_freq")
    #feature_names.append("z_dominant_freq")



    #x.append(_compute_peak_count_feature(window))
    #feature_names.append("x_peak_count")
    #feature_names.append("y_peak_count")
    #feature_names.append("z_peak_count")



    x.append(getEntropy(window))
    feature_names.append("x_entropy")
    feature_names.append("y_entropy")
    feature_names.append("z_entropy")

    feature_vector = np.concatenate(x, axis=0) # convert the list of features to a single 1-dimensional vector
    return feature_names, feature_vector