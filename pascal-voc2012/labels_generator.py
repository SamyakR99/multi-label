################# TRAIN ######################

import numpy as np

total_classes = 20
train_labels = voc2012.train_labels  # Assuming you have a train_labels variable
total_train_lab = np.empty((0, total_classes))  # Initialize empty array outside the loop

for idx, lab in enumerate(train_labels):
    lab_array = np.zeros((total_classes))
    unique_pixel_values = np.unique(lab)
    unique_pixel_values = unique_pixel_values[unique_pixel_values > 0]  # Filter out values <= 0
    lab_array[unique_pixel_values - 1] = 1
    total_train_lab = np.vstack((total_train_lab, lab_array))  # Stack lab_array to total_train_lab

np.save('total_train_lab.npy', total_train_lab)

################# VAL ######################

import numpy as np

total_classes = 20
val_labels = voc2012.val_labels
total_val_lab = np.empty((0, total_classes))  # Initialize empty array outside the loop

for idx, lab in enumerate(labels):
    lab_array = np.zeros((total_classes))
    unique_pixel_values = np.unique(lab)
    unique_pixel_values = unique_pixel_values[unique_pixel_values > 0]  # Filter out values <= 0
    lab_array[unique_pixel_values - 1] = 1
    total_val_lab = np.vstack((total_val_lab, lab_array))  # Stack lab_array to total_val_lab
np.save('total_val_lab.npy', total_val_lab)
