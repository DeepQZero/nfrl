import csv
import numpy as np


def csv_saver(filepath: str, outpath: str) -> None:
    """Pickles a .csv file and saves it to a specified path."""
    print('PICKLING FILE: ' + filepath)
    all_data = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            all_data.append(row)
    np_all_data = np.array(all_data)
    np.save(outpath, np_all_data)
    print('PICKLING FILE: ' + filepath + ' IS COMPLETE')


def np_loader(filepath: str) -> np.array:
    """Loads numpy file. This code is primarily for personal reference."""
    return np.load(filepath)


def save_all_csvs() -> None:
    """Pickles all of the .csv data files from the Kaggle competition."""
    for week in range(1, 18):
        filepath = 'raw_data/csv_files/week' + str(week) + '.csv'
        outpath = 'raw_data/numpy_data/week' + str(week) + '.npy'
        csv_saver(filepath, outpath)
    csv_saver('raw_data/csv_files/plays.csv',
              'raw_data/numpy_data/plays.npy')
    csv_saver('raw_data/csv_files/players.csv',
              'raw_data/numpy_data/players.npy')
    csv_saver('raw_data/csv_files/games.csv',
              'raw_data/numpy_data/games.npy')


if __name__ == '__main__':
    save_all_csvs()
