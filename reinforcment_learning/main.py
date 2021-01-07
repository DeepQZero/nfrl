from agent import Agent
import numpy as np

play_mat = np.load('../data/win_probability/pbp_data/all_pbp_data.npy')
X = play_mat[:, 2:9].astype(np.float32)
rows = X.shape[0]


def_agent = Agent()
off_agent = Agent()


def_agent.train(X[0], 10, 0.55, 0.57)