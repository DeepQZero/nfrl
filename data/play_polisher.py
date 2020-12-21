import pickle
import datetime
import numpy as np
from multiprocessing import Pool


class PlayPolisher:
    def __init__(self, game_id, play_id):
        self.game_id = game_id
        self.play_id = play_id
        self.game = None
        self.play = None
        self.stamps = None
        self.ball_snap = None
        self.los = None
        self.player_dict = None
        self.play_direction = None
        self.off_players = None
        self.def_players = None
        self.football = None
        self.init_everything()

    def init_everything(self):
        game_dict = pickle.load(open("raw_data/game_dicts/" + self.game_id + ".p", "rb"))
        self.game = game_dict['game']
        self.get_play(game_dict['plays'])
        self.get_play_stamps(game_dict['stamps'])

    def get_play_stamps(self, game_stamps):
        self.stamps = []
        for stamp in game_stamps:
            if stamp[16] == self.play_id:
                self.stamps.append(stamp)

    def get_play(self, game_plays):
        for play in game_plays:
            if play[1] == self.play_id:
                self.play = play
                break

    def polish(self):
        if not self.check_ball_snap():
            # print(self.game_id, self.play_id, "FAILED BALL SNAP")
            return None
        if not self.check_los():
            # print(self.game_id, self.play_id, "FAILED LOS")
            return None
        if not self.check_player_stamps():
            # print(self.game_id, self.play_id, "FAILED STAMP")
            return None
        if not self.check_player_teams():
            # print(self.game_id, self.play_id, "FAILED TEAM")
            return None
        if not self.check_seconds():
            # print(self.game_id, self.play_id, "FAILED TIME")
            return None
        if not self.check_direction():
            # print(self.game_id, self.play_id, "FAILED DIRECTION")
            return None
        if not self.check_sides():
            # print(self.game_id, self.play_id, "FAILED SIDES")
            return None
        # print("POLISHED")
        self.pickle_game()

    def pickle_game(self):
        play_dict = {'game_id': self.game_id,
                     'play_id': self.play_id,
                     'game': self.game,
                     'play': self.play,
                     'stamps': self.stamps,
                     'ball_snap': self.ball_snap,
                     'los': self.los,
                     'player_dict': self.player_dict,
                     'play_direction': self.play_direction,
                     'off_players': self.off_players,
                     'def_players': self.def_players,
                     'football': self.football}
        outfile = 'polished_data/play_dicts/' + self.game_id + '-' + self.play_id + '.p'
        pickle.dump(play_dict, open(outfile, 'wb'))

    def check_ball_snap(self):
        for stamp in self.stamps:
            if stamp[8] == 'ball_snap':
                self.ball_snap = int(stamp[13])
                return True
        return False

    def check_los(self):
        los = self.play[19]
        if not self.check_float(los):
            return False
        else:
            self.los = float(los)
            return True

    def check_float(self, potential_float):
        try:
            float(potential_float)
            return True
        except ValueError:
            return False

    def check_player_stamps(self):
        players = set(stamp[9] for stamp in self.stamps)
        player_dict = {player: [] for player in players}
        for stamp in self.stamps:
            player_dict[stamp[9]].append((int(stamp[13]), float(stamp[1]), float(stamp[2]),
                                          stamp[14], self.to_sec(stamp[0])))
        for player in players:
            player_dict[player].sort()
        sizes = [len(player_dict[player]) for player in players]
        for size in sizes:
            if size != sizes[0]:
                return False
        self.player_dict = player_dict
        return True

    def check_player_teams(self):
        for player in self.player_dict.keys():
            teams = [stamp[3] for stamp in self.player_dict[player]]
            for team in teams:
                if team != teams[0]:
                    return False
        return True

    def check_seconds(self):
        time_tuples = []
        for player in self.player_dict.keys():
            times = []
            for stamp in self.player_dict[player]:
                times.append(stamp[4])
            time_tuples.append(tuple(times))
        for time_tuple in time_tuples:
            if time_tuple != time_tuples[0]:
                return False
        return True

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

    def check_direction(self):
        self.play_direction = self.stamps[0][17]
        if self.play_direction not in {'left', 'right'}:
            return False
        for play in self.stamps:
            if play[17] != self.play_direction:
                return False
        return True

    def check_sides(self):
        home_players, away_players, self.football = set(), set(), set()
        for player in self.player_dict:
            team = self.player_dict[player][0][3]
            if team == 'football':
                self.football.add(player)
            elif team == 'home':
                home_players.add(player)
            else:
                away_players.add(player)
        if len(self.football) != 1:
            return False
        home_x = np.mean([self.player_dict[player][self.ball_snap - 1][1]
                          for player in home_players])
        away_x = np.mean([self.player_dict[player][self.ball_snap - 1][1]
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


def pooler(play):
    game_id, play_id = play[0], play[1]
    polisher = PlayPolisher(game_id, play_id)
    polisher.polish()


if __name__ == "__main__":
    play_matrix = np.load('../data/raw_data/numpy_data/plays.npy')
    plays = play_matrix[1:, :]
    p = Pool(6)
    p.map(pooler, plays)
    p.close()
    p.join()




