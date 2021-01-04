import numpy as np
import datetime
import os


def check_unique_game_ids():
    game_matrix = np.load('../data/raw_data/numpy_data/games.npy')
    print(len(np.unique(game_matrix[1:, 0])))
    print(len(game_matrix[1:, 0]))


def check_game_id_lengths():
    game_matrix = np.load('../data/raw_data/numpy_data/games.npy')
    for id in game_matrix[1:, 0]:
        if len(id) != 10:
            print(id)


def check_unique_play_game_ids():
    play_matrix = np.load('../data/raw_data/numpy_data/plays.npy')
    all_play_game_ids = []
    for play in play_matrix[1:, :]:
        all_play_game_ids.append(play[0]+play[1])
    print(len(np.unique(all_play_game_ids)))
    print(len(play_matrix[1:, 1]))


def check_play_id_lengths():
    play_matrix = np.load('../data/raw_data/numpy_data/plays.npy')
    for id in play_matrix[1:, 1]:
        if len(id) < 2:
            print(id)


def check_correct_game_id_in_play():
    game_matrix = np.load('../data/raw_data/numpy_data/games.npy')
    all_game_ids = set(game_matrix[1:, 0])
    play_matrix = np.load('../data/raw_data/numpy_data/plays.npy')
    for play in play_matrix[1:, :]:
        if play[0] not in all_game_ids:
            print(play[0])


def check_correct_game_id_play_id_in_stamp():
    game_matrix = np.load('../data/raw_data/numpy_data/games.npy')
    all_game_ids = set(game_matrix[1:, 0])
    play_matrix = np.load('../data/raw_data/numpy_data/plays.npy')
    all_play_ids = set(play_matrix[1:, 1])
    for i in range(1, 18):
        print('WEEK: ' + str(i))
        week_matrix = np.load('../data/raw_data/numpy_data/week' + str(i) + '.npy')
        for stamp in week_matrix[1:, :]:
            game_id, play_id = stamp[15], stamp[16]
            if game_id not in all_game_ids or play_id not in all_play_ids:
                print(stamp)


def check_los():
    all_bad = []
    play_matrix = np.load('../data/raw_data/numpy_data/plays.npy')
    for play in play_matrix[1:, :]:
        if not check_float(play[19]):
            print(play[19])
            all_bad.append(play[0] + play[1])
            print('NOT FLOAT: ', len(play[19]))
        else:
            los = float(play[19])
            if float(los) > 110 or float(los) < 10:
                all_bad.append(play[0] + play[1])
                print('NOT IN RANGE: ', los)
    print('NUMBER BAD: ', len(all_bad))


def check_float(potential_float):
    try:
        float(potential_float)
        return True
    except ValueError:
        return False


def check_int(potential_float):
    try:
        int(potential_float)
        return True
    except ValueError:
        return False


def check_home_visitor_teams():
    game_matrix = np.load('../data/raw_data/numpy_data/games.npy')
    all_home = set(game_matrix[1:, 3])
    all_away = set(game_matrix[1:, 4])
    print(len(all_home), len(all_away))


def check_possession_team():
    game_matrix = np.load('../data/raw_data/numpy_data/games.npy')
    all_home = set(game_matrix[1:, 3])
    play_matrix = np.load('../data/raw_data/numpy_data/plays.npy')
    for possession_team in play_matrix[1:, 6]:
        if possession_team not in all_home:
            print(possession_team)


def check_play_direction():
    for i in range(1, 18):
        print('WEEK: ' + str(i))
        week_matrix = np.load(
            '../data/raw_data/numpy_data/week' + str(i) + '.npy')
        for stamp in week_matrix[1:, :]:
            if stamp[17] not in {'left', 'right'}:
                print(stamp)


def check_team():
    for i in range(1, 18):
        print('WEEK: ' + str(i))
        week_matrix = np.load(
            '../data/raw_data/numpy_data/week' + str(i) + '.npy')
        for stamp in week_matrix[1:, :]:
            if stamp[14] not in {'home', 'away', 'football'}:
                print(stamp)


def check_frame_id():
    for i in range(1, 18):
        print('WEEK: ' + str(i))
        week_matrix = np.load(
            '../data/raw_data/numpy_data/week' + str(i) + '.npy')
        for stamp in week_matrix[1:, :]:
            if not check_int(stamp[13]):
                print(stamp)
            else:
                frame_id = int(stamp[13])
                if frame_id > 1000 or frame_id < 1:
                    print(frame_id)


def check_positions():
    for i in range(1, 18):
        week_games = []
        print('WEEK: ' + str(i))
        week_matrix = np.load(
            '../data/raw_data/numpy_data/week' + str(i) + '.npy')
        for stamp in week_matrix[1:, :]:
            if not check_float(stamp[1]) or not check_float(stamp[2]):
                print(stamp)
            else:
                x, y = float(stamp[1]), float(stamp[2])
                if not 0.0 <= x <= 120.0 or not 0.0 <= y <= 54.0:
                    week_games.append(stamp[15] + stamp[16])
        print(np.unique(week_games))
        print(len(np.unique(week_games)))


def check_max_positions():
    x_vals = []
    y_vals = []
    for i in range(1, 18):
        print('WEEK: ' + str(i))
        week_matrix = np.load(
            '../data/raw_data/numpy_data/week' + str(i) + '.npy')
        for stamp in week_matrix[1:, :]:
            x_vals.append(float(stamp[1]))
            y_vals.append(float(stamp[2]))
    print(min(x_vals), max(x_vals))
    print(min(y_vals), max(y_vals))


def check_times():
    def to_sec(stamp: str) -> float:
        """Converts a timestamp to a seconds format."""
        year = int(stamp[0:4])
        month = int(stamp[5:7])
        day = int(stamp[8:10])
        hour = int(stamp[11:13])
        minute = int(stamp[14:16])
        second = int(stamp[17:19])
        microsecond = int(stamp[20:23]) * 1000
        date = datetime.datetime(year, month, day, hour, minute, second,
                                 microsecond)
        delta = date - datetime.datetime(1970, 1, 1)
        return delta.total_seconds()
    all_times = []
    for i in range(1, 18):
        week_games = []
        print('WEEK: ' + str(i))
        week_matrix = np.load(
            '../data/raw_data/numpy_data/week' + str(i) + '.npy')
        for stamp in week_matrix[1:, :]:
            time = to_sec(stamp[0])
            if time < 100 or time > 100000000000000:
                print(time)
            else:
                all_times.append(time)
    print(min(all_times), max(all_times), len(all_times))


def check_stamps_sizes():
    the_sum = 0
    for i in range(1, 18):
        print('WEEK: ' + str(i))
        week_matrix = np.load(
            '../data/raw_data/numpy_data/week' + str(i) + '.npy')
        for stamp in week_matrix[1:, :]:
            the_sum += 1
    print(the_sum)


def check_plays_sizes():
    the_sum = 0
    play_matrix = np.load('../data/raw_data/numpy_data/plays.npy')
    for play in play_matrix[1:, :]:
        the_sum += 1
    print(the_sum)


def count_folder_files():
    path, dirs, files = next(os.walk("play_pictures/def"))
    file_count = len(files)
    print(file_count)


if __name__ == "__main__":
    check_los()
