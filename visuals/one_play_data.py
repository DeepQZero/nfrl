import numpy as np


def get_play_data(playid, rows):
    off_players, def_players = get_players(rows)
    off_routes, def_routes = get_routes(rows, off_players, def_players)


def get_players(rows):
    off_team, def_team = det_off_team(rows)
    off_players, def_players = set(), set()
    for row in rows:
        if row[12] == 'football':
            break
        if row[12] == off_team:
            off_players.add(row[9])
        else:
            def_players.add(row[9])


def det_off_team(rows):
    for row in rows:
        if row[12] == 'QB':
            if row[14] == 'home':
                return 'home', 'away'
            else:
                return 'away', 'home'
    raise Exception('NO QB BAD')


def get_routes(rows, off_players, def_players):
    return None, None