import csv
import numpy as np


def csv_to_np_converter(filepath: str, outpath: str) -> None:
    """Converts a .csv file to a .npy matrix and saves to a specified path."""
    print('CONVERTING FILE: ' + filepath)
    all_data = []
    with open(filepath, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            all_data.append(row)
    np_all_data = np.array(all_data)
    np.save(outpath, np_all_data)
    print('CONVERTING FILE: ' + filepath + ' IS COMPLETE')


def np_loader(filepath: str) -> np.array:
    """Loads numpy file. This code is primarily for personal reference."""
    return np.load(filepath)


def convert_all_kaggle_csv_to_np() -> None:
    """Converts all of the .csv Kaggle data to .npy matrices."""
    for week in range(1, 18):
        filepath = 'raw_data/csv_files/kaggle/week' + str(week) + '.csv'
        outpath = 'raw_data/numpy_data/kaggle/week' + str(week) + '.npy'
        csv_to_np_converter(filepath, outpath)
    csv_to_np_converter('raw_data/csv_files/kaggle/plays.csv',
                        'raw_data/numpy_data/kaggle/plays.npy')
    csv_to_np_converter('raw_data/csv_files/kaggle/players.csv',
                        'raw_data/numpy_data/kaggle/players.npy')
    csv_to_np_converter('raw_data/csv_files/kaggle/games.csv',
                        'raw_data/numpy_data/kaggle/games.npy')


def convert_all_nfl_scrapr_to_np() -> None:
    """Converts some of the .csv NFL scrapR data to .npy matrices."""
    for year in range(2009, 2019):
        filepath = 'raw_data/csv_files/nfl_scrapr/reg_games_' \
                   + str(year) + '.csv'
        outpath = 'raw_data/numpy_data/nfl_scrapr/reg_games_' \
                  + str(year) + '.npy'
        csv_to_np_converter(filepath, outpath)


if __name__ == '__main__':
    convert_all_kaggle_csv_to_np()
    convert_all_nfl_scrapr_to_np()
