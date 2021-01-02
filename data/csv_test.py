import csv
import numpy as np
import pickle


def csv_pickler(filename: str) -> None:
    """Pickles a .csv file and saves it to a specified path."""
    print('PICKLING FILE: ' + filename)
    all_data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        counter = 0
        for row in reader:
            print(row)
            for idx, elem in enumerate(row):
                if elem == 'play_type':
                    print(idx)
            # print(row[7])
            # if row[8] == '50':
            #     print(row[7], row[8])
            # print(row[24])
            counter += 1
            # if row[1] == '2018090600':
            #     print(row)
            # timeout = row[45]
            # if timeout == '1':
            #     print(row)
            if counter == 10:
                break
            # all_data.append(row)
    print(counter)
    # np_all_data = np.array(all_data)
    # np.save(outname, np_all_data)
    # print('PICKLING FILE: ' + filename + ' IS COMPLETE')


def unpickler(filename) -> np.array:
    """Unpickles numpy file. This code is primarily for personal reference."""
    return np.load(filename)


def pickle_all_csv_files() -> None:
    """Pickles all of the .csv data files."""
    for week in range(1, 18):
        filename = 'raw_data/csv_files/week' + str(week) + '.csv'
        outname = 'raw_data/numpy_data/week' + str(week) + '.npy'
        csv_pickler(filename, outname)
    csv_pickler('raw_data/csv_files/plays.csv', 'raw_data/numpy_data/plays.npy')
    csv_pickler('raw_data/csv_files/players.csv', 'raw_data/numpy_data/players.npy')
    csv_pickler('raw_data/csv_files/games.csv', 'raw_data/numpy_data/games.npy')


def create_game_dict():
    game_dict = {}
    for i in range(11):
        year = str(2009 + i)
        x = np.load('raw_data/numpy_data/reg_games_' + year + '.npy')
        for row in x[1:, :]:
            game_dict[row[1]] = {'id': row[1],
                                 'home': row[2],
                                 'away': row[3],
                                 'home_score': row[8],
                                 'away_score': row[9]}
    return game_dict


def check_int(potential_float):
    try:
        int(potential_float)
        return True
    except ValueError:
        return False


def create_stats(play_stats, game_stats):
    home_team = play_stats[2]
    away_team = play_stats[3]
    if home_team != game_stats['home'] or away_team != game_stats['away']:
        return None
    if play_stats[4] not in {home_team, away_team}:
        return None
    else:
        poss_team = play_stats[4]
    if poss_team == game_stats['home']:
        poss_fin_score, def_fin_score = game_stats['home_score'], game_stats['away_score']
    else:
        poss_fin_score, def_fin_score = game_stats['away_score'], game_stats['home_score']
    if poss_fin_score > def_fin_score:
        poss_est = 1
    elif poss_fin_score < def_fin_score:
        poss_est = 0
    else:
        poss_est = 0.5
    if check_int(play_stats[17]) and int(play_stats[17]) in {1, 2, 3, 4}:
        quarter = int(play_stats[17])
    else:
        return None
    if check_int(play_stats[18]) and int(play_stats[18]) in {1, 2, 3, 4}:
        down = int(play_stats[18])
    else:
        return None
    if check_int(play_stats[52]):
        poss_score = int(play_stats[52])
    else:
        return None
    if check_int(play_stats[53]):
        def_score = int(play_stats[53])
    else:
        return None
    if check_int(play_stats[22]):
        yards_to_go = int(play_stats[22])
    else:
        return None
    if check_int(play_stats[8]):
        yardline_100 = int(play_stats[8])
    else:
        return None
    if play_stats[25] in {'pass', 'run', 'punt', 'kickoff', 'field_goal'}:
        play_type = play_stats[25]
    else:
        return None
    if play_stats[7] in {home_team, away_team, 'MID'}:
        field_side = play_stats[7]
    else:
        return None
    if field_side == 'MID':
        relative = 50
    elif poss_team == home_team:
        if field_side == home_team:
            relative = max(100 - yardline_100, yardline_100)
        else:
            relative = min(100 - yardline_100, yardline_100)
    else:
        if field_side == home_team:
            relative = min(100 - yardline_100, yardline_100)
        else:
            relative = max(100 - yardline_100, yardline_100)
    if check_int(play_stats[12]):
        time_left = int(play_stats[12])
    else:
        return None
    all_stats = {'home_team': home_team,
                 'away_team': away_team,
                 'poss_team': poss_team,
                 'quarter': quarter,
                 'poss_est': poss_est,
                 'poss_score': poss_score,
                 'def_score': def_score,
                 'down': down,
                 'yards_to_go': yards_to_go,
                 'play_type': play_type,
                 'absolute': relative,
                 'time_left': time_left}
    return all_stats


if __name__ == '__main__':
    game_dict = create_game_dict()
    play_dict = {}
    filename = 'raw_data/csv_files/big-play-by-play.csv'
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        counter = 0
        for row in reader:
            if counter != 0:
                all_stats = create_stats(row, game_dict[row[1]])
                if all_stats != None:
                    play_dict[(row[1], row[0])] = all_stats
            counter += 1
            if (counter % 1000) == 0:
                print(counter)
    outfile = 'polished_data/big_play_dicts/dicts.p'
    pickle.dump(play_dict, open(outfile, 'wb'))