import numpy as np
import datetime
import pickle


class GamePolisher:
    """Sorts AWS timestamp data into standalone dictionaries; polishes data."""
    def __init__(self) -> None:
        """Loads small data files."""
        self.game_matrix = np.load('raw_data/numpy_data/games.npy')
        self.play__matrix = np.load('raw_data/numpy_data/plays.npy')

    def polish_all_games(self) -> None:
        """Main function. Polishes all game data sequentially."""
        for idx, game_id in enumerate(np.unique(self.game_matrix[1:, 0])):
            print('PICKLING GAME NUMBER: ' + str(idx+1))
            self.polish_one_game(game_id)

    def find_game(self, game_id: str) -> str:
        """Finds a raw game data give game id."""
        for game_data in self.game_matrix[1:, :]:
            if game_id == game_data[0]:
                return game_data
        raise Exception('No matching game_id')

    def polish_one_game(self, game_id: str) -> None:
        """Polishes data from a single game."""
        game_data = self.find_game(game_id)
        plays = self.get_game_plays(game_id, game_data[5])
        game_data = {'id': game_data[0],
                     'home': game_data[3],
                     'away': game_data[4],
                     'week': game_data[5],
                     'plays': plays}
        outfile = 'polished_data/game_dicts/' + str(game_id) + '.p'
        pickle.dump(game_data, open(outfile, 'wb'))

    def init_play_dicts(self, game_id: str) -> dict:
        """Initializes a dictionary of plays and corresponding AWS stamps
         for current game."""
        plays = {}
        for play_data in self.play__matrix[1:, :]:
            if play_data[0] == game_id:
                play_id = play_data[1]
                plays[play_id] = {'possession': play_data[6], 'stamps': []}
        return plays

    def get_game_plays(self, game_id: str, week: str) -> dict:
        """Places AWS stamps into corresponding plays for current game."""
        plays = self.init_play_dicts(game_id)
        week_matrix = np.load('raw_data/numpy_data/week' + str(week) + '.npy')
        for stamp in week_matrix[1:]:
            if stamp[15] == game_id:
                new_stamp = np.concatenate([[self.to_sec(stamp[0])], stamp[1:]])
                play_id = stamp[16]
                plays[play_id]['stamps'].append(new_stamp)
        return plays

    @staticmethod
    def to_sec(stamp: str) -> float:
        """Converts a timestamp to a seconds format."""
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


if __name__ == "__main__":
    pickler = GamePolisher()
    pickler.polish_all_games()
