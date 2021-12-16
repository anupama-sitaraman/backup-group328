"""
This is the script used to train an activity recognition 
classifier on accelerometer data.

"""

import os
import sys
import numpy as np
from sklearn.tree import export_graphviz
from features import extract_features
from util import slidingWindow, reorient, reset_vars
import pickle
from sklearn import model_selection
from sklearn import tree
from sklearn.metrics import confusion_matrix

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
class_names = ["running", "walking", "speed walking", "sitting","dancing"]

print("Extracting features and labels for window size {} and step size {}...".format(window_size, step_size))
sys.stdout.flush()

X = []
Y = []

for i,window_with_timestamp_and_label in slidingWindow(data, window_size, step_size):
    window = window_with_timestamp_and_label[:,1:-1]   
    feature_names, x = extract_features(window)
    X.append(x)
    Y.append(window_with_timestamp_and_label[10, -1])
    
X = np.asarray(X)
Y = np.asarray(Y)
n_features = len(X)
    
print("Finished feature extraction over {} windows".format(len(X)))
print("Unique labels found: {}".format(set(Y)))
print("\n")
sys.stdout.flush()

# %%---------------------------------------------------------------------------
#
#		                Train & Evaluate Classifier
#
# -----------------------------------------------------------------------------


# TODO: split data into train and test datasets using 10-fold cross validation

"""
TODO: iterating over each fold, fit a decision tree classifier on the training set.
Then predict the class labels for the test set and compute the confusion matrix
using predicted labels and ground truth values. Print the accuracy, precision and recall
for each fold.
"""
predicted_arr =[]
correct_labels_arr = []
accuracy_arr = []
cv = model_selection.KFold(n_splits=10, random_state=None, shuffle=True)
for train_index, test_index in cv.split(X):
    tr = tree.DecisionTreeClassifier(criterion="entropy", max_depth=15)
    #tr.fit(X, class_names) 
    tr.fit(X[train_index], Y[train_index])
    predicted = tr.predict(X[test_index])
    predicted_arr.append(predicted)
    correct_labels_arr.append(map(Y.__getitem__, test_index))

    


# TODO: calculate and print the average accuracy, precision and recall values over all 10 folds
#confusion matrix
confusion_matricies = []
for i in range(len(predicted_arr)):
    conf = confusion_matrix(list(correct_labels_arr[i]),predicted_arr[i])
    confusion_matricies.append(conf)
accuracy_arr = []
precision_arr_running = []
precision_arr_walking = []
precision_arr_speed_walking= []
precision_arr_sitting = []
precision_arr_dancing = []
recall_arr_running = []
recall_arr_walking = []
recall_arr_speed_walking = []
recall_arr_sitting = []
recall_arr_dancing = []
#fix
for c in confusion_matricies:
    TP_running = c[0][0]
    TP_walking = c[1][1]
    TP_speed_walking = c[2][2]
    TP_sitting = c[3][3]
    TP_dancing=  c[4][4]
    #####fix####
    FN_running = c[1][0] + c[2][0] + c[3][0] + c[4][0]
    FN_walking = c[2][1] + c[0][1] + c[3][1] + c[4][1]
    FN_speed_walking = c[1][2] + c[0][2] + c[3][2] + c[4][2]
    FN_sitting =  c[0][3] + c[1][3] + c[2][3] + c[4][3]
    FN_dancing =  c[0][4] + c[1][4] + c[2][4] + c[3][4]
    FP_running = c[0][1] + c[0][2] + c[0][3] + c[0][4]
    FP_walking = c[1][0] + c[1][2] + c[1][3] + c[1][4]
    FP_speed_walking = c[2][0] + c[2][1] + c[2][3] + c[2][4]
    FP_sitting = c[3][0] + c[3][1] + c[3][2] + c[3][4]
    FP_dancing = c[4][0] + c[4][1] + c[4][2] + c[4][3]
    
    accuracy_arr.append((TP_dancing + TP_sitting + TP_walking  + TP_running + TP_speed_walking) / (TP_dancing + TP_sitting + TP_walking  + TP_running + TP_speed_walking + FP_dancing + FP_sitting + FP_walking  + FP_running + FP_speed_walking + FN_dancing + FN_sitting + FN_walking  + FN_running + FN_speed_walking))
    precision_arr_running.append(TP_running / (TP_running + FP_running))
    precision_arr_walking.append(TP_walking / (TP_walking + FP_walking))
    precision_arr_speed_walking.append(TP_speed_walking / (TP_speed_walking + FP_speed_walking))
    precision_arr_sitting.append(TP_sitting / (TP_sitting + FP_sitting))
    precision_arr_dancing.append(TP_dancing / (TP_dancing + FP_dancing))
    
    recall_arr_running.append(TP_running / (TP_running + FN_running))
    recall_arr_walking.append(TP_walking / (TP_walking + FN_walking))
    recall_arr_speed_walking.append(TP_speed_walking / (TP_speed_walking + FN_speed_walking))
    recall_arr_sitting.append(TP_sitting / (TP_sitting + FN_sitting))
    recall_arr_dancing.append(TP_dancing / (TP_dancing+ FN_dancing))
    
print("average recall for running {} ".format(np.average(recall_arr_running)))
print("average recall for walking {} ".format(np.average(recall_arr_walking)))
print("average recall for  speed walking {} ".format(np.average(recall_arr_speed_walking)))
print("average recall for sitting {} ".format(np.average(recall_arr_sitting)))
print("average recall for dancing {} ".format(np.average(recall_arr_dancing)))
print("average precision for running {} ".format(np.average(precision_arr_running)))
print("average precision for walking {} ".format(np.average(precision_arr_walking)))
print("average precision for speed walking {} ".format(np.average(precision_arr_speed_walking)))
print("average precision for sitting {} ".format(np.average(precision_arr_sitting)))
print("average precision for dancing {} ".format(np.average(precision_arr_dancing)))
print("average accuracy {} ".format(np.average(accuracy_arr)))

    

# TODO: train the decision tree classifier on entire dataset


# TODO: Save the decision tree visualization to disk - replace 'tree' with your decision tree and run the below line
export_graphviz(tr, out_file='tree.dot', feature_names = feature_names)

# TODO: Save the classifier to disk - replace 'tree' with your decision tree and run the below line
#Maybe? export_graphviz(tree, out_file='classifier.dot', class_names = class_names)
with open('classifier.pickle', 'wb') as f:
    pickle.dump(tr, f)