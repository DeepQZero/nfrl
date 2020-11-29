import csv
import numpy as np


def pickler(filename, outname):
    all_data = []
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            all_data.append(row)
    final_data = np.array(all_data)
    np.save(outname, final_data)


# e.g. Windows: pickler(r'raw_data\week1.csv', r'numpy_data\week1.npy')


def unpickler(filename):
    return np.load(filename)


# e.g. Windows: unpickler(r'numpy_data\week1.npy')



