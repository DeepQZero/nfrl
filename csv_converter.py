import csv
import numpy as np


def week_pickler(filename='week1.csv', outname='week1'):
    all_data = []
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            all_data.append(row)
    final_data = np.array(all_data)
    np.save(outname, final_data)


def play_pickler(filename='plays.csv', outname='plays'):
    all_data = []
    with open(filename, newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            all_data.append(row)
    final_data = np.array(all_data)
    np.save(outname, final_data)


play_pickler()