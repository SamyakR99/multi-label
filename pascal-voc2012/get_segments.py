import numpy as np
import matplotlib.pyplot as plt

############ FOR ONE IMAGE AND ITS LABEL ############

image = batch_train_images[0]
label = batch_train_labels[0]

unique_pixel_values = np.unique(label)
unique_pixel_values = unique_pixel_values[unique_pixel_values > 0]
num_unique_pixels = len(unique_pixel_values)
print(unique_pixel_values)
segmented_masks = []

fig, axes = plt.subplots(1, num_unique_pixels, figsize=(15, 3))

for i, pixel_value in enumerate(unique_pixel_values):
    # Create an image with only the current unique pixel value
    unique_label = np.zeros_like(label)
    unique_label[label == pixel_value] = pixel_value
    segmented_masks.append(unique_label)
    # Plot the unique image
    axes[i].imshow(unique_label, cmap='gray')
    axes[i].set_title(f"Pixel Value: {pixel_value}")
    axes[i].axis('off')

plt.show()

