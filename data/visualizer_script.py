from data.play_visualizer import PlayVisualizer
from data.conversions import unpickler


play__matrix = unpickler(r'numpy_data\plays.npy')
for idx, play in enumerate(play__matrix[1:, :]):
    if idx % 100 == 0:
        print(idx)
    game_id, play_id = play[0], play[1]
    for team in ['def', 'off']:
        vis = PlayVisualizer(game_id, play_id, team)
        vis.attempt_plot()

