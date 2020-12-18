from visuals.play_visualizer import PlayVisualizer
from data.conversions import unpickler


play__matrix = unpickler(r'../data/raw_data/numpy_data/plays.npy')
for idx, play in enumerate(play__matrix[2270:, :]):
    if idx % 10 == 0:
        print(idx)
    game_id, play_id = play[0], play[1]
    for team in ['def', 'off']:
        vis = PlayVisualizer(game_id, play_id, team)
        vis.attempt_plot()

