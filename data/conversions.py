import csv
import numpy as np


def pickler(filename: str, outname: str) -> None:
    """Pickles a file and saves it to a specified path."""
    print('PICKLING FILE: ' + filename)
    all_data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            all_data.append(row)
    final_data = np.array(all_data)
    np.save(outname, final_data)
    print('PICKLING FILE: ' + filename + ' IS COMPLETE')


def unpickler(filename) -> np.array:
    """Unpickles numpy file. This code is primarily for personal reference."""
    return np.load(filename)


def pickle_all_csv_files() -> None:
    """Pickles all of the .csv data files"""
    for week in range(1, 18):
        filename = 'raw_data/csv_files/week' + str(week) + '.csv'
        outname = 'raw_data/numpy_data/week' + str(week) + '.npy'
        pickler(filename, outname)
    pickler('raw_data/csv_files/plays.csv', 'raw_data/numpy_data/plays.npy')
    pickler('raw_data/csv_files/players.csv', 'raw_data/numpy_data//players.npy')
    pickler('raw_data/csv_files/games.csv', 'raw_data/numpy_data//games.npy')


if __name__ == '__main__':
    pickle_all_csv_files()
