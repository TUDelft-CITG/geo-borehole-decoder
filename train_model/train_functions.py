import cv2
import numpy as np
import os


def load_image_resize(data_dir, labels, img_size):
    """Load the image, create labels, resize the image"""
    data = []

    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[
                    ..., ::-1
                ]  # convert BGR to RGB format
                resized_arr = cv2.resize(
                    img_arr, (img_size, img_size)
                )  # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)


def data_augment(data, norm_factor, img_size):
    """Create features and labels, normalise the data."""
    x_values = []
    y_values = []

    for feature, label in data:
        x_values.append(feature)
        y_values.append(label)

    # Normalize the data
    x_values = np.array(x_values) / norm_factor

    x_values.reshape(-1, img_size, img_size, 1)
    y_values = np.array(y_values)
    return x_values, y_values
