import numpy as np
import pickle


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

from agent import Agent
import numpy as np


play_mat = np.load('../data/raw_data/numpy_data/kaggle/plays.npy')
labels = play_mat[0]
data = play_mat[1:]


play_mat = np.load('../data/raw_data/numpy_data/kaggle/plays.npy')


def_clusters = np.load('../data/clustering/plays/def/clusterings.npy')
off_clusters = np.load('../data/clustering/plays/off/clusterings.npy')


game_dict = {}
games = np.load('../data/raw_data/numpy_data/kaggle/games.npy')
for row in games[1:]:
    game_dict[row[0]] = {'game_id': row[0],
                         'home': row[4],
                         'visitor': row[5]}


def bad_row(row):
    bad_set = {'', 'NA'}
    if row[17] in bad_set:
        return True
    return False


def convert_row_to_state(row):
    game_id = str(row[0])
    play_id = str(row[1])
    home_team = game_dict[game_id]['home']
    quarter = int(row[3])
    down = int(row[4])
    first_down_yards = int(row[5])
    home_score = int(row[17])
    visitor_score = int(row[16])
    possession_team = row[6]
    play_result = int(row[23])
    if home_team == possession_team:
        off_score, def_score = home_score, visitor_score
    else:
        def_score, off_score = visitor_score, home_score
    game_clock = row[18]
    time = 3600 - 900 * quarter + 60 * int(game_clock[:2]) + int(game_clock[3:5])
    yard_line_side = str(row[8])
    yard_line_number = int(row[9])
    if possession_team == yard_line_side:
        yard = max(100 - yard_line_number, yard_line_number)
    else:
        yard = min(100 - yard_line_number, yard_line_number)
    return [quarter, off_score, def_score, down,
            first_down_yards, yard, time, play_result]


good_data = {}
for row in data:
    if bad_row(row):
        continue
    else:
        new_row = convert_row_to_state(row)
        good_data[(row[0], row[1])] = new_row

def_labels = {}
for row in def_clusters:
    game_id = str(int(row[0]))
    play_id = str(int(row[1]))
    label = int(row[-1])
    def_labels[(game_id, play_id)] = label

off_labels = {}
for row in off_clusters:
    game_id = str(int(row[0]))
    play_id = str(int(row[1]))
    label = int(row[-1])
    off_labels[(game_id, play_id)] = label

datas = []
targets = []
for key in good_data.keys():
    if key in def_labels and key in off_labels:
        data = good_data[key]
        off_label = off_labels[key]
        def_label = def_labels[key]
        new_data = data[0:5] + [data[6]] + [off_label] + [def_label]
        target = data[7]
        targets.append(target)
        datas.append(new_data)
X = np.array(datas)
y = targets


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
# print(X.shape)

log_reg = SGDClassifier(loss='log')
log_reg.fit(X, y)
pass_classes = log_reg.classes_
# print(X[0])
# for i in range(25):
#     print(log_reg.predict_proba(X[i].reshape(1, -1)))


play_mat = np.load('../data/win_probability/pbp_data/all_pbp_data.npy')
play_samples = play_mat[:, 2:8].astype(np.float32)
rows = play_samples.shape[0]
# print(X[1])


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 6)
        self.fc2 = nn.Linear(6, 5)
        self.fc3 = nn.Linear(5, 4)
        self.fc4 = nn.Linear(4, 3)
        self.fc5 = nn.Linear(3, 2)
        self.fc6 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = torch.sigmoid(self.fc6(x))
        return x


class Env:
    def __init__(self):
        self.state = None
        self.model = Net()
        self.model.load_state_dict(
            torch.load('../data/win_probability/snapshots/predictor_1'))

    def get_new_state(self):
        while True:
            chance = np.random.randint(0, rows)
            data = play_samples[chance]
            if data[0] < 4.0:
                self.state = data
                return data

    def reset(self):
        return self.get_new_state()

    def step(self, off_action, def_action):
        the_play = np.concatenate([self.state, [off_action], [def_action]]).reshape(1, -1)
        transformed_play = scaler.transform(the_play)
        probs = log_reg.predict_proba(transformed_play)
        yardage = np.random.choice(pass_classes, p=list(probs[0]))
        time = self.state[5]
        off_score = self.state[1]
        def_score = self.state[2]
        quarter = self.state[0]
        # print(yardage)
        if yardage > self.state[4]:
            down = 1
            first_down = 10
        else:
            down = min(self.state[3] + 1, 4)
            first_down = min(self.state[4] - yardage, 10)
        new_state = np.array([quarter, off_score, def_score, down, first_down, time])
        # print(self.state, new_state)
        old_percentage = self.model(torch.from_numpy(self.state).float()).item()
        new_percentage = self.model(torch.from_numpy(new_state).float()).item()
        off = new_percentage - old_percentage
        defen = old_percentage - new_percentage
        # print(old_percentage, new_percentage)
        self.state = self.get_new_state()
        return self.state, off, defen


# [quarter, off_score, def_score, down,
# first_down_yards, yard, time, play_result]


env = Env()
state = env.reset()
# env.step(3, 2)

rewards = []
def_agent = Agent()
off_agent = Agent()
epsilon = 0.9999
for i in range(100000):
    if i % 10 == 0:
        if i > 200:
            avg_rew = float(np.mean(rewards[-1000:]))
            print(i, str(round(epsilon, 2)), str(round(avg_rew, 3)))
    # def_act = def_agent.act(state, epsilon)
    def_act = 3
    off_act = off_agent.act(state, epsilon)
    print(off_act)
    new_state, off, defen = env.step(off_act, def_act)
    # def_agent.train(state, def_act, defen)
    off_agent.train(state, off_act, off)
    rewards.append(off)
    epsilon = max(epsilon - 0.0001, 0.02)
    state = new_state

