import numpy as np
import conversions as conv
import datetime
import pickle


class GamePickler:
    def __init__(self):
        self.game_matrix = conv.unpickler(r'numpy_data\games.npy')
        self.play__matrix = conv.unpickler(r'numpy_data\plays.npy')

    def pickle_all_games(self):
        for idx, game_id in enumerate(np.unique(self.game_matrix[1:, 0])):
            print(idx+1)
            self.pickle_one_game(game_id)

    def find_game(self, game_id):
        for game in self.game_matrix[1:, :]:
            if game_id == game[0]:
                return game
        raise Exception('No matching game_id')

    def pickle_one_game(self, game_id):
        game = self.find_game(game_id)
        plays = self.get_game_plays(game_id, game[5])
        game_data = {'id': game[0], 'home': game[3], 'away': game[4],
                     'week': game[5], 'plays': plays}
        pickle.dump(game_data, open(r'game_data\game' + str(game_id) + r'.p', 'wb'))

    def init_play_dicts(self, game_id):
        plays = {}
        for play in self.play__matrix[1:, :]:
            if play[0] == game_id:
                plays[play[1]] = {'possession': play[6], 'stamps': []}
        return plays

    def get_game_plays(self, game_id, week):
        plays = self.init_play_dicts(game_id)
        week_matrix = conv.unpickler(r'numpy_data\week' + str(week) + '.npy')
        for stamp in week_matrix[1:]:
            if stamp[15] == game_id:
                new_stamp = np.concatenate([[self.to_sec(stamp[0])], stamp[1:]])
                plays[stamp[16]]['stamps'].append(new_stamp)
        return plays

    @staticmethod
    def to_sec(stamp):
        year = int(stamp[0:4])
        month = int(stamp[5:7])
        day = int(stamp[8:10])
        hour = int(stamp[11:13])
        minute = int(stamp[14:16])
        second = int(stamp[17:19])
        microsecond = int(stamp[20:23]) * 1000
        date = datetime.datetime(year, month, day, hour, minute, second, microsecond)
        delta = date - datetime.datetime(1970, 1, 1)
        return delta.total_seconds()


pickler = GamePickler()
pickler.pickle_all_games()