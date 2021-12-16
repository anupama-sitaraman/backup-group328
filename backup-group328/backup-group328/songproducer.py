import os
import sys
import numpy as np
from sklearn.tree import export_graphviz
from stepcountingfeatures import extract_features
from util import slidingWindow, reorient, reset_vars
import pickle
from sklearn import model_selection
from sklearn import tree
from sklearn.metrics import confusion_matrix
import stepcountingfeatures

# %%---------------------------------------------------------------------------
#
#		                 Load Data From Disk
#
# -----------------------------------------------------------------------------

print("Loading data...")
sys.stdout.flush()
data_file = 'anu_jogging_walking_raine_sitting_stairs.csv'
data = np.genfromtxt(data_file, delimiter=',')
print("Loaded {} raw labelled activity data samples.".format(len(data)))
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                    Pre-processing
#
# -----------------------------------------------------------------------------

print("Reorienting accelerometer data...")
#sys.stdout.flush()
reset_vars()
reoriented = np.asarray([reorient(data[i,1], data[i,2], data[i,3]) for i in range(len(data))])
reoriented_data_with_timestamps = np.append(data[:,0:1],reoriented,axis=1)
data = np.append(reoriented_data_with_timestamps, data[:,-1:], axis=1)

# %%---------------------------------------------------------------------------
#
#		                Extract Features & Labels
#
# -----------------------------------------------------------------------------

window_size = 20
step_size = 20

# sampling rate should be about 25 Hz; you can take a brief window to confirm this
n_samples = 1000
time_elapsed_seconds = (data[n_samples,0] - data[0,0]) / 1000
sampling_rate = n_samples / time_elapsed_seconds

# TODO: list the class labels that you collected data for in the order of label_index (defined in collect-labelled-data.py)
#class_names = ["running", "walking", "speed walking", "sitting","dancing"]

#print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
#sys.stdout.flush()

X = []
Y = []

for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,1:-1]   
    #feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])
    
#X = np.asarray(X)
#Y = np.asarray(Y)
#n_features = len(X)

songName = getSong(filtSignal(window)[4])
print(songName)
    
#print("Finished feature extraction over {} windows".format(len(X)))
#print("Unique labels found: {}".format(set(Y)))
#print("\n")
sys.stdout.flush()