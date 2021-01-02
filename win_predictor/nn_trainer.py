import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


games = []
game_dict = pickle.load(open("../data/polished_data/big_play_dicts/dicts.p", "rb"))
for key in game_dict.keys():
    game = game_dict[key]
    game_info = []
    game_info.append(game['quarter'])
    game_info.append(game['poss_score'] - game['def_score'])
    game_info.append(game['down'])
    game_info.append(game['yards_to_go'])
    game_info.append(game['absolute'])
    game_info.append(game['time_left'])
    game_info.append(game['poss_est'])
    games.append(game_info)
final_info = np.array(games)
final_info = final_info.astype(np.float32)

# print(final_info.shape)

X = final_info[:, :6]
y = final_info[:, 6]
y = y.reshape(-1, 1)

# print(X.shape)
# print(y.shape)


train, test = train_test_split(list(range(X.shape[0])), test_size=.1)

batch_size = 8


class PrepareData(Dataset):
    def __init__(self, X, y):
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = self.scaler.transform(X)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def scale_trans(self, new_data):
        return self.scaler.transform(new_data)

    def inv_trans(self, new_data):
        return self.scaler.inverse_transform(new_data)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


ds = PrepareData(X, y=y)

train_set = DataLoader(ds, batch_size=batch_size,
                       sampler=SubsetRandomSampler(train))
test_set = DataLoader(ds, batch_size=batch_size,
                      sampler=SubsetRandomSampler(test))


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


def train():
    the_net = Net()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(the_net.parameters(), lr=1e-3)

    for epoch in range(10):
        running_loss = []
        for batch_idx, (data, targets) in enumerate(train_set):
            optimizer.zero_grad()
            outputs = the_net(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
            if batch_idx % 1000 == 0:
                print('Epoch: ', epoch, ', Batch: ', batch_idx, ', Loss: ', np.mean(running_loss))
        torch.save(the_net.state_dict(), 'snapshots/predictor_' + str(epoch+1))


# def check():
#     model = Net()
#     model.load_state_dict(torch.load('snapshots/predictor_10'))
#
#     # game_info.append(game['quarter'])
#     # game_info.append(game['poss_score'])
#     # game_info.append(game['def_score'])
#     # game_info.append(game['down'])
#     # game_info.append(game['yards_to_go'])
#     # game_info.append(game['absolute'])
#     # game_info.append(game['time_left'])
#     # game_info.append(game['poss_est'])
#
#     fake_data = ds.scale_trans(np.array([[3, 20, 3, 1, 10, 75, 1000]], dtype=np.float32))
#     fake_data = torch.tensor(fake_data)
#     output = model(fake_data)
#     print(output)


def observe():
    model = Net()
    model.load_state_dict(torch.load('snapshots/predictor_10'))
    for batch_idx, (data, lab) in enumerate(train_set):
        old_data = ds.inv_trans(data)
        old_data = old_data.astype(int)
        for x in old_data:
            print(x)
        output = model(data)
        print(output)
        break


def test():
    the_net = Net()
    criterion = nn.MSELoss()
    the_net.load_state_dict(torch.load('snapshots/predictor_10'))
    running_loss = []
    for batch_idx, (data, targets) in enumerate(test_set):
        outputs = the_net(data)
        loss = criterion(outputs, targets)
        running_loss.append(loss.item())
    print('Loss: ', np.mean(running_loss))


def plot():
    game_plays = []
    for key in game_dict.keys():
        if key[0] == '2018111900':
            game_plays.append(key)
    game_info = []
    for key in game_plays:
        game_info.append(get_stuff(game_dict[key]))
    the_net = Net()
    the_net.load_state_dict(torch.load('snapshots/predictor_10'))
    predictions = []
    for idx, data in enumerate(game_info):
        outputs = the_net(data)
        prediction = outputs[0][0].item()
        key = game_plays[idx]
        stats = game_dict[key]
        home_team = stats['home_team']
        away_team = stats['away_team']
        poss_team = stats['poss_team']
        if home_team != poss_team and away_team != poss_team:
            print("BAD BAD BAD!!!!!!!!!!")
        else:
            if home_team == poss_team:
                predictions.append(1.0 - prediction)
            else:
                predictions.append(prediction)
    times = []
    for key in game_plays:
        times.append(3600 - game_dict[key]['time_left'])
    print(times, predictions)

    plt.plot(times, predictions)
    plt.axhline(0.5)
    plt.axhline(0.75)
    plt.axhline(0.25)
    plt.show()


def get_stuff(game):
    game_info = []
    game_info.append(game['quarter'])
    game_info.append(game['poss_score'] - game['def_score'])
    game_info.append(game['down'])
    game_info.append(game['yards_to_go'])
    game_info.append(game['absolute'])
    game_info.append(game['time_left'])
    game_info = np.array([game_info]).astype(np.float32)
    game_info = ds.scale_trans(game_info)
    game_info = torch.from_numpy(game_info)
    return game_info


if __name__ == '__main__':
    plot()
