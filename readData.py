import numpy as np
import struct
from array import array
def readData():
    def read_mnist(images_filepath, labels_filepath):
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError(f'Label file magic number mismatch, expected 2049, got {magic}')
            labels = array("B", file.read())
        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError(f'Image file magic number mismatch, expected 2051, got {magic}')
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols]).reshape(28, 28)
            images.append(img)
        return list(zip(images, labels))

    # Set file paths for your dataset
    training_images_filepath = 'train-images-idx3-ubyte/train-images-idx3-ubyte'
    training_labels_filepath = 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'
    test_images_filepath = 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
    test_labels_filepath = 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'

    # Read and store as (image, label) pairs
    train_data = read_mnist(training_images_filepath, training_labels_filepath)
    test_data = read_mnist(test_images_filepath, test_labels_filepath)

    return train_data, test_data