import pickle


class PlayFormatter:
    """Formats polished data for a single play to be used for visualization."""
    def __init__(self) -> None:
        """No fields currently needed."""
        pass

    def format_play(self, game_id: str, play_id: str) -> dict:
        game_file = '../data/polished_data/game_dicts/' + game_id + '.p'
        game_data = pickle.load(open(game_file, 'rb'))
        return self.arrange_data(game_data['plays'][play_id]['stamps'],
                                 game_data['home'],
                                 game_data['away'],
                                 game_data['plays'][play_id]['possession'])

    def det_snap_time(self, stamps: list) -> float:
        """Determines time of ball snap."""
        for stamp in stamps:
            if stamp[8] == 'ball_snap':
                return stamp[0]
        raise Exception('No ball_snap')

    def det_players(self, stamps: list) -> set:
        """Determines all players in a specific play."""
        players = set()
        for stamp in stamps:
            if stamp[14] != 'football':
                players.add((stamp[9], stamp[14]))
        return players

    def init_player_dict(self, stamps: list, home: str, away: str, poss: str) -> dict:
        """Initializes a dictionary of players and corresponding AWS stamps."""
        player_dict = {}
        players = self.det_players(stamps)
        for player_id, player_team in players:
            side = self.det_side(player_team, home, away, poss)
            player_dict[player_id] = {'id': player_id,
                                      'side': side,
                                      'positions': []}
        return player_dict

    def det_side(self, player_team: str, home: str, away: str, poss: str) -> str:
        """Determines which side (offense/defense) is on the field."""
        if player_team == 'home' and home == poss:
            return 'off'
        elif player_team == 'away' and away == poss:
            return 'off'
        if player_team == 'away' and home == poss:
            return 'def'
        elif player_team == 'home' and away == poss:
            return 'def'
        else:
            raise Exception('Unable to determine possession')

    def arrange_data(self, stamps: list, home: str, away: str, poss: str) -> dict:
        """Creates dictionary of players and associated sorted timestamps."""
        snap_time = self.det_snap_time(stamps)
        player_dict = self.init_player_dict(stamps, home, away, poss)
        for stamp in stamps:
            if stamp[14] != 'football' and stamp[0] >= snap_time:
                player_dict[stamp[9]]['positions'].append((stamp[0],
                                                           float(stamp[1]),
                                                           float(stamp[2])))
        for values in player_dict.values():
            values['positions'].sort()
        return player_dict


if __name__ == "__main__":
    formatter = PlayFormatter()
    routes = formatter.format_play('2018090600', '75')
    print(routes)
