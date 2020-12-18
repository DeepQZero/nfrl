import numpy as np
import datetime
import pickle


# polished_data = pickle.load(open(r'polished_data\game2018090600.p', 'rb'))
# print(polished_data.keys())
# plays = polished_data['plays']
# for key in plays.keys():
#     stamps = plays[key]['stamps']
#     num_stamps = len(stamps)
#     players = set()
#     for stamp in stamps:
#         players.add(stamp[9])
#     num_players = len(players)
#     print(num_stamps, num_players, num_stamps / num_players)


# polished_data = pickle.load(open(r'polished_data\game2018090600.p', 'rb'))
# print(polished_data.keys())
# plays = polished_data['plays']
# for key in plays.keys():
#     print(plays[key].keys())
#     stamps = plays[key]['stamps']
#     all_stamps = []
#     for stamp in stamps:
#         all_stamps.append((float(stamp[0]), float(stamp[1]), float(stamp[2])))
#     all_stamps.sort()
#     # print(all_stamps)
