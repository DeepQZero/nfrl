import numpy as np
import pickle


class GameOrganizer:
    """Sorts AWS timestamp data into standalone dictionaries; polishes data."""
    def __init__(self) -> None:
        """Loads small data files."""
        self.game_matrix = np.load('raw_data/numpy_data/games.npy')
        self.play_matrix = np.load('raw_data/numpy_data/plays.npy')

    def polish_all_games(self) -> None:
        """Main function. Polishes all play data sequentially."""
        for idx, game in enumerate(self.game_matrix[1:, :]):
            print('PICKLING GAME NUMBER: ' + str(idx+1))
            self.polish_one_game(game)

    def polish_one_game(self, game: np.ndarray) -> None:
        """Polishes data from a single game."""
        play_data = {'game': game,
                     'plays': self.get_plays(game),
                     'stamps': self.get_stamps(game)}
        outfile = 'raw_data/game_dicts/' + game[0] + '.p'
        pickle.dump(play_data, open(outfile, 'wb'))

    def get_stamps(self, game: np.ndarray) -> list:
        week_matrix = np.load('raw_data/numpy_data/week' + str(game[5]) + '.npy')
        return [stamp for stamp in week_matrix[1:] if stamp[15] == game[0]]

    def get_plays(self, game: np.ndarray) -> list:
        """Places AWS stamps into corresponding plays for current game."""
        return [play for play in self.play_matrix[1:] if game[0] == play[0]]


if __name__ == "__main__":
    organizer = GameOrganizer()
    organizer.polish_all_games()
