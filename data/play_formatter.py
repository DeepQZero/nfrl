import numpy as np
import datetime
import pickle


class PlayFormatter:
    def __init__(self):
        pass

    def format_play(self, game_id, play_id):
        game_file = r'polished_data\game' + game_id + '.p'
        game_data = pickle.load(open(game_file, 'rb'))
        return self.arrange_data(game_data['plays'][play_id]['stamps'],
                                 game_data['home'], game_data['away'],
                                 game_data['plays'][play_id]['possession'])

    def det_snap_time(self, stamps):
        for stamp in stamps:
            if stamp[8] == 'ball_snap':
                return stamp[0]
        raise Exception('No ball_snap')

    def init_player_dict(self, stamps, home, away, poss):
        player_dict = {}
        players = set()
        for stamp in stamps:
            if stamp[14] != 'football':
                players.add((int(stamp[9]), stamp[14]))
        for player_id, player_team in players:
            side = self.det_off(player_team, home, away, poss)
            player_dict[player_id] = {'id': player_id, 'side': side, 'positions': []}
        return player_dict

    def det_off(self, player_team, home, away, poss):
        if player_team == 'home' and home == poss:
            return 'off'
        elif player_team == 'away' and away == poss:
            return 'off'
        else:
            return 'def'

    def arrange_data(self, stamps, home, away, poss):
        snap_time = self.det_snap_time(stamps)
        player_dict = self.init_player_dict(stamps, home, away, poss)
        for stamp in stamps:
            if stamp[14] != 'football' and stamp[0] >= snap_time:
                player_dict[int(stamp[9])]['positions'].append((stamp[0], float(stamp[1]),
                                                            float(stamp[2])))
        for values in player_dict.values():
            values['positions'].sort()
        return player_dict


# formatter = PlayFormatter()
# routes = formatter.format_play('2018090600', '75')
