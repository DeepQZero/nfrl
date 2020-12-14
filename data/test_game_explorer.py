import numpy as np
import datetime
import pickle


game_data = pickle.load(open(r'game_data\game2018090600.p', 'rb'))
print(game_data.keys())
plays = game_data['plays']
for key in plays.keys():
    stamps = plays[key]['stamps']
    num_stamps = len(stamps)
    players = set()
    for stamp in stamps:
        players.add(stamp[9])
    num_players = len(players)
    print(num_stamps, num_players, num_stamps / num_players)

