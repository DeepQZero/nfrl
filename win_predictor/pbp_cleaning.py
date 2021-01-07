import numpy as np
import csv
import pickle


def final_score_dict_creator():
    final_score_dict = {}
    for year in range(2009, 2019):
        outpath = '../data/raw_data/numpy_data/nfl_scrapr/reg_games_' + \
                   str(year) + '.npy'
        year_data = np.load(outpath)
        for row in year_data[1:, :]:
            game_id = row[1]
            row_dict = {'home_team': row[2],
                        'away_team': row[3],
                        'home_score': row[8],
                        'away_score': row[9]}
            final_score_dict[game_id] = row_dict
    outfile = '../data/dictionaries/final_scores/dict.p'
    pickle.dump(final_score_dict, open(outfile, 'wb'))


def possession_dict_creator():
    final_score_dict = {}
    for year in range(2009, 2019):
        outpath = '../data/raw_data/csv_files/nfl_scrapr/reg_pbp_' + \
                   str(year) + '.csv'
        with open(outpath, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for idx, row in enumerate(reader):
                if idx != 0:
                    game_id = row[1]
                    play_id = row[0]
                    row_dict = {'home_team': row[2],
                                'away_team': row[3],
                                'posteam': row[4],
                                'time_left': row[12]}
                    final_score_dict[(game_id, play_id)] = row_dict
    outfile = '../data/dictionaries/possessions/dict.p'
    pickle.dump(final_score_dict, open(outfile, 'wb'))


def pbp_appender():
    all_pbp = []
    for year in range(2009, 2018):
        new_data = pbp_one_year(year)
        all_pbp += new_data
    all_pbp = np.array(all_pbp)
    outpath = '../data/win_probability/pbp_data/all_pbp_data.npy'
    np.save(outpath, all_pbp)


def pbp_one_year(year: int):
    pbp_path = '../data/raw_data/csv_files/nfl_scrapr/reg_pbp_' + \
                str(year) + '.csv'
    game_dict_file = '../data/dictionaries/final_scores/dict.p'
    final_score_dict = pickle.load(open(game_dict_file, "rb"))
    year_pbp_data = []
    with open(pbp_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            play_data = []
            if idx != 0:
                if check_bad_row(row):
                    continue
                # play_dict['play_id'] = int(row[0])
                # play_dict['game_id'] = int(row[1])
                # play_dict['quarter'] = int(row[17])
                # play_dict['posteam_score'] = int(row[52])
                # play_dict['defteam_score'] = int(row[53])
                # play_dict['down'] = int(row[18])
                # play_dict['first_down_yards'] = int(row[22])
                # play_dict['absolute'] = det_yard_to_go(row)
                # play_dict['time_left'] = int(row[12])
                # play_dict['det_reward'] = det_reward(row, final_score_dict)
                play_data.append(int(row[0]))
                play_data.append(int(row[1]))
                play_data.append(int(row[17]))
                play_data.append(int(row[52]))
                play_data.append(int(row[53]))
                play_data.append(int(row[18]))
                play_data.append(int(row[22]))
                # play_data.append(det_yard_to_go(row))
                play_data.append(int(row[12]))
                play_data.append(det_reward(row, final_score_dict))
                year_pbp_data.append(play_data)
    # x = np.array(year_pbp_data)
    # print(len(np.unique(x[:, 1])))
    # print(x.shape)
    return year_pbp_data


def check_bad_row(row):
    bad_set = {'', 'NA'}
    if row[0] in bad_set:
        return True
    if row[1] in bad_set:
        return True
    if row[17] in bad_set:
        return True
    if row[52] in bad_set:
        return True
    if row[53] in bad_set:
        return True
    if row[18] in bad_set:
        return True
    if row[22] in bad_set:
        return True
    if row[8] in bad_set:
        return True
    if row[12] in bad_set:
        return True
    return False


def det_yard_to_go(row):
    yardline_100 = int(row[8])
    posteam = row[4]
    side_of_field = row[7]
    if posteam == side_of_field:
        return min(100 - yardline_100, yardline_100)
    else:
        return max(100 - yardline_100, yardline_100)


def det_reward(row, final_score_dict):
    game_id = row[1]
    home_team = row[2]
    posteam = row[4]
    home_score = int(final_score_dict[game_id]['home_score'])
    away_score = int(final_score_dict[game_id]['away_score'])
    if home_score == away_score:
        return 0.5
    elif posteam == home_team:
        return 1 if home_score > away_score else 0
    else:
        return 1 if away_score > home_score else 0


def find_field(field: str):
    pbp_path = '../data/raw_data/csv_files/nfl_scrapr/reg_pbp_' + \
               str(2009) + '.csv'
    with open(pbp_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            print(row)
            for idx, attr in enumerate(row):
                if attr == field:
                    print('INDEX OF ' + field + " IS: " + str(idx))
            break


def print_some_fields(field_idx: int):
    pbp_path = '../data/raw_data/csv_files/nfl_scrapr/reg_pbp_' + \
               str(2009) + '.csv'
    with open(pbp_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            print(row[field_idx])
            if idx == 25:
                break


if __name__ == '__main__':
    final_score_dict_creator()
    # find_field('side_of_field')
    # print_some_fields(7)
    pbp_appender()
    # possession_dict_creator()
