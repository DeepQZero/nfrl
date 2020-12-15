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


# for i in range(1, 18, 1):
#     print(i)
#     week = i
#     filename = r'raw_data\week' + str(week) + '.csv'
#     outname = r'numpy_data\week' + str(week) + '.npy'
#     pickler(filename, outname)


# old_matrix = unpickler(r'numpy_data\week1.npy')
# for i in range(2, 18, 1):
#     print(i)
#     new_matrix = unpickler(r'numpy_data\week' + str(i) + '.npy')
#     old_matrix = np.concatenate((old_matrix, new_matrix), axis=0)
# np.save(r'numpy_data\all_weeks.npy', old_matrix)


# pickler(r'raw_data\plays.csv', r'numpy_data\plays.npy')
