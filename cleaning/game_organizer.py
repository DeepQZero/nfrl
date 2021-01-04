import numpy as np
import pickle


class GameOrganizer:
    """Sorts AWS timestamp data into standalone dictionaries per game."""
    def __init__(self) -> None:
        """Loads small data files."""
        self.game_mat = np.load('raw_data/numpy_data/games.npy')
        self.play_mat = np.load('raw_data/numpy_data/plays.npy')

    def organize_all_games(self) -> None:
        """Sorts and creates game dictionary holding all data about game."""
        for idx, gamestamp in enumerate(self.game_mat[1:, :]):
            print('PICKLING GAME NUMBER: ' + str(idx+1))
            self.organize_one_game(gamestamp)

    def organize_one_game(self, gamestamp: np.ndarray) -> None:
        """Collect and organize data from a single game.
           Uses game id as identifier in filename."""
        game_data = {'game': gamestamp,
                     'plays': self.get_playstamps(gamestamp),
                     'stamps': self.get_timestamps(gamestamp)}
        outfile = 'raw_data/game_dicts/' + gamestamp[0] + '.p'
        pickle.dump(game_data, open(outfile, 'wb'))

    def get_playstamps(self, game: np.ndarray) -> list:
        """Gets every playstamp that shares same game id."""
        return [stamp for stamp in self.play_mat[1:] if game[0] == stamp[0]]

    def get_timestamps(self, game: np.ndarray) -> list:
        """Gets every timestamp that shares same game id."""
        week_mat = np.load('raw_data/numpy_data/week' + str(game[5]) + '.npy')
        return [stamp for stamp in week_mat[1:] if stamp[15] == game[0]]


if __name__ == "__main__":
    organizer = GameOrganizer()
    organizer.organize_all_games()
