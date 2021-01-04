import pickle
import datetime
import numpy as np
from multiprocessing import Pool
from data_utils import check_float
import operator


class PlayPolisher:
    """Creates a polished play dictionary that is ready for image creation."""
    def __init__(self, game_id: str, play_id: str) -> None:
        """Sets all fields and loads necessary game data."""
        self.game_id = game_id
        self.play_id = play_id
        self.gamestamp = None
        self.playstamp = None
        self.timestamps = None
        self.snap_frame = None
        self.los = None
        self.player_info = None
        self.play_direction = None
        self.off_players = None
        self.def_players = None
        self.football = None
        self.load_data()

    def load_data(self) -> None:
        """Loads game dictionary and fills corresponding fields."""
        game_dict_file = "raw_data/dictionaries/games/" + self.game_id + ".p"
        game_dict = pickle.load(open(game_dict_file, "rb"))
        self.gamestamp = game_dict['game']
        self.playstamp = self.get_playstamp(game_dict['plays'])
        self.timestamps = self.get_timestamps(game_dict['stamps'])
        self.construct_player_dict()

    def get_playstamp(self, gamestamps: list) -> np.ndarray:
        """Gets playstamp corresponding to play id."""
        for play in gamestamps:
            if play[1] == self.play_id:
                return play

    def get_timestamps(self, timestamps: list) -> list:
        """Gets all timestamps of current play."""
        return [stamp for stamp in timestamps if stamp[16] == self.play_id]

    def construct_player_dict(self) -> None:
        """Constructs a dictionary of players with associated timestamps."""
        player_ids = set(stamp[9] for stamp in self.timestamps)
        player_info = {player: [] for player in player_ids}
        for stamp in self.timestamps:
            stamp_info = self.det_player_timestamps(stamp)
            player_info[stamp[9]].append(stamp_info)
        self.player_info = player_info
        self.sort_player_dict()

    def sort_player_dict(self) -> None:
        """Sorts the timestamps of each player."""
        for player in self.player_info:
            self.player_info[player].sort(key=operator.itemgetter('frame_id'))

    def det_player_timestamps(self, stamp: str) -> dict:
        """Places relevant timestamp data into dictionary form."""
        return {'frame_id': int(stamp[13]),
                'x_pos': float(stamp[1]),
                'y_pos': float(stamp[2]),
                'team': stamp[14],
                'sec': self.to_sec(stamp[0])}

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
        date = datetime.datetime(year, month, day, hour, minute, second,
                                 microsecond)
        delta = date - datetime.datetime(1970, 1, 1)
        return delta.total_seconds()

    def polish(self) -> None:
        """Creates dictionary of play information and pickles it.
        If some information isn't available, does nothing."""
        if not self.check_ball_snap():
            return None
        if not self.check_los():
            return None
        if not self.check_player_stamps():
            return None
        if not self.check_player_teams():
            return None
        if not self.check_seconds():
            return None
        if not self.check_direction():
            return None
        if not self.check_sides():
            return None
        self.pickle_game()

    def pickle_game(self) -> None:
        """Pickles game after all fields have been set."""
        play_dict = {'game_id': self.game_id,
                     'play_id': self.play_id,
                     'gamestamp': self.gamestamp,
                     'playstamp': self.playstamp,
                     'timestamps': self.timestamps,
                     'snap_frame': self.snap_frame,
                     'los': self.los,
                     'player_info': self.player_info,
                     'play_direction': self.play_direction,
                     'off_players': self.off_players,
                     'def_players': self.def_players,
                     'football': self.football}
        outfile = 'dictionaries/plays/' + \
                  self.game_id + '-' + self.play_id + '.p'
        pickle.dump(play_dict, open(outfile, 'wb'))

    def check_ball_snap(self) -> bool:
        """Determines the frame id of the ball snap."""
        for timestamp in self.timestamps:
            if timestamp[8] == 'ball_snap':
                self.snap_frame = int(timestamp[13])
                return True
        return False

    def check_los(self) -> bool:
        """Determines the line of scrimmage. Integer from 10 to 110."""
        los = self.playstamp[19]
        if not check_float(los):
            return False
        else:
            self.los = float(los)
            return True

    def check_player_stamps(self) -> bool:
        """Determines if each player has the same number of timestamps."""
        sizes = [len(self.player_info[player]) for player in self.player_info]
        for size in sizes:
            if size != sizes[0]:
                return False
        return True

    def check_player_teams(self) -> bool:
        """Determines if player has same team on each timestamp."""
        for player in self.player_info.keys():
            teams = [stamp['team'] for stamp in self.player_info[player]]
            for team in teams:
                if team != teams[0]:
                    return False
        return True

    def check_seconds(self) -> bool:
        """Determines if each player's timestamps occurred at the same time."""
        time_tuples = []
        for player in self.player_info.keys():
            times = [stamp['sec'] for stamp in self.player_info[player]]
            time_tuples.append(tuple(times))
        for time_tuple in time_tuples:
            if time_tuple != time_tuples[0]:
                return False
        return True

    def check_direction(self) -> bool:
        """Determines if each play has same direction."""
        self.play_direction = self.timestamps[0][17]
        if self.play_direction not in {'left', 'right'}:
            return False
        for play in self.timestamps:
            if play[17] != self.play_direction:
                return False
        return True

    def check_sides(self):
        """Determines if all players have a single team."""
        home_players, away_players, self.football = self.get_off_def()
        if len(self.football) != 1:
            return False
        home_x = np.mean([self.player_info[player][self.snap_frame - 1]['x_pos']
                          for player in home_players])
        away_x = np.mean([self.player_info[player][self.snap_frame - 1]['x_pos']
                          for player in away_players])
        if home_x > self.los and away_x > self.los:
            return False
        elif home_x < self.los and away_x < self.los:
            return False
        elif home_x < self.los < away_x:
            if self.play_direction == 'left':
                self.off_players = away_players
                self.def_players = home_players
            else:
                self.off_players = home_players
                self.def_players = away_players
        else:
            if self.play_direction == 'left':
                self.off_players = home_players
                self.def_players = away_players
            else:
                self.off_players = away_players
                self.def_players = home_players
        return True

    def get_off_def(self) -> tuple:
        """Determines offense, defense, and football status for each player."""
        home_players, away_players, football = set(), set(), set()
        for player in self.player_info:
            team = self.player_info[player][0]['team']
            if team == 'football':
                football.add(player)
            elif team == 'home':
                home_players.add(player)
            else:
                away_players.add(player)
        return home_players, away_players, football


def pooler(playstamp: np.ndarray) -> None:
    """Converts all plays with valid info to pickle objects."""
    game_id, play_id = playstamp[0], playstamp[1]
    polisher = PlayPolisher(game_id, play_id)
    polisher.polish()


if __name__ == "__main__":
    play_matrix = np.load('/raw_data/numpy_data/kaggle/plays.npy')
    plays = play_matrix[1:, :]
    p = Pool(6)
    p.map(pooler, plays)
    p.close()
    p.join()