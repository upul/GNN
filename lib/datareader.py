import numpy as np
import os, struct
from array import array as pyarray

def load_mnist(x_loc, y_loc, digits=np.arange(10)):
    """ doc """
    flbl = open(y_loc, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(x_loc, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = np.zeros((N, rows, cols), dtype=np.float64)
    labels = np.zeros((N, 1), dtype=np.int8)
    for i in range(len(ind)):
        images[i] = np.array(img[ ind[i] * rows * cols : (ind[i] + 1) * rows * cols ]).reshape((rows, cols)) / 255.0
        labels[i] = lbl[ind[i]]

    return images, labels
