import numpy as np


def save(inputs, outputs, fname):
    if len(inputs.shape) == 1:
        inputs = inputs.reshape(inputs.shape[0], 1)
    if len(outputs.shape) == 1:
        outputs = outputs.reshape(outputs.shape[0], 1)

    res = np.hstack([inputs, outputs])

    np.savetxt(fname, res, fmt="%.3f", delimiter=",", header="x,p")


def load(fname):
    data = np.loadtxt(fname, skiprows=1, delimiter=",", dtype=np.float32)
    inputs, outputs = data[:, 0], data[:, 1:].T
    return inputs, outputs
