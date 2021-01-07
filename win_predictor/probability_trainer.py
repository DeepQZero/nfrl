import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt


from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import csv


play_mat = np.load('../data/win_probability/pbp_data/all_pbp_data.npy')

print(len(np.unique(play_mat[:, 1])))

X = play_mat[:, 2:8].astype(np.float32)
y = play_mat[:, 8].astype(np.float32)
y = y.reshape(-1, 1)

print(X.shape)
print(y.shape)

train, test = train_test_split(list(range(X.shape[0])), test_size=.1)

batch_size = 8


# TODO Set Seed

class PrepareData(Dataset):
    def __init__(self, X, y):
        # self.scaler = StandardScaler()
        # self.scaler.fit(X)
        # X = self.scaler.transform(X)
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
                print('Epoch: ', epoch,
                      ', Batch: ', batch_idx,
                      ', Loss: ', np.mean(running_loss))
        outfile = '../data/win_probability/snapshots/predictor_' + str(epoch+1)
        torch.save(the_net.state_dict(), outfile)


def check():
    model = Net()
    model.load_state_dict(torch.load('../data/win_probability/snapshots/predictor_1'))

    # play_dict['quarter'] = int(row[17])
    # play_dict['posteam_score'] = int(row[52])
    # play_dict['defteam_score'] = int(row[53])
    # play_dict['down'] = int(row[18])
    # play_dict['first_down_yards'] = int(row[22])
    # play_dict['absolute'] = det_yard_to_go(row)
    # play_dict['time_left'] = int(row[12])
    # play_dict['det_reward'] = det_reward(row, final_score_dict)

    fake_data = np.array([[1, 0, 0, 1, 10, 3600]], dtype=np.float32)
    fake_data = torch.tensor(fake_data)
    output = model(fake_data)
    print(output)

    fake_data = np.array([[1, 0, 0, 2, 10, 3600]], dtype=np.float32)
    fake_data = torch.tensor(fake_data)
    output = model(fake_data)
    print(output)


def observe():
    model = Net()
    model.load_state_dict(torch.load('../data/win_probability/snapshots/predictor_10'))
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
    the_net.load_state_dict(torch.load('../data/win_probability/snapshots/predictor_10'))
    running_loss = []
    for batch_idx, (data, targets) in enumerate(test_set):
        outputs = the_net(data)
        loss = criterion(outputs, targets)
        running_loss.append(loss.item())
    print('Loss: ', np.mean(running_loss))


def plot():
    game = '2018092313'
    year = 2018
    predictions, times = [], []
    the_net = Net()
    the_net.load_state_dict(torch.load('../data/win_probability/snapshots/predictor_10'))
    poss_dict_path = '../data/dictionaries/possessions/dict.p'
    poss_dict = pickle.load(open(poss_dict_path, "rb"))
    pbp_path = '../data/raw_data/csv_files/nfl_scrapr/reg_pbp_' + \
                str(year) + '.csv'
    with open(pbp_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for idx, row in enumerate(reader):
            if row[1] == game:
                is_bad = check_bad_row(row)
                if is_bad:
                    continue
                play_data = []
                play_data.append(int(row[17]))
                play_data.append(int(row[52]))
                play_data.append(int(row[53]))
                play_data.append(int(row[18]))
                play_data.append(int(row[22]))
                play_data.append(det_yard_to_go(row))
                play_data.append(int(row[12]))
                print(play_data)
                fake_data = np.array([play_data], dtype=np.float32)
                outputs = the_net(torch.tensor(fake_data))
                print(outputs)
                prediction = outputs[0].item()
                key = (row[1], row[0])
                stats = poss_dict[key]
                home_team = stats['home_team']
                away_team = stats['away_team']
                poss_team = stats['posteam']
                time_left = stats['time_left']
                if home_team != poss_team and away_team != poss_team:
                    print("BAD BAD BAD!!!!!!!!!!")
                else:
                    if home_team == poss_team:
                        predictions.append(1.0 - prediction)
                        times.append(3600 - int(time_left))
                    else:
                        predictions.append(prediction)
                        times.append(3600 - int(time_left))

    print(predictions)
    plt.plot(times, predictions)
    plt.axhline(0.5)
    plt.axhline(0.75)
    plt.axhline(0.25)
    plt.show()


def check_bad_row(row):
    bad_set = {'', 'NA'}
    if row[0] in bad_set:
        return True
    if row[1] in bad_set:
        return True
    if row[17] in bad_set:
        return True
    if row[52] in bad_set:
        return True
    if row[53] in bad_set:
        return True
    if row[18] in bad_set:
        return True
    if row[22] in bad_set:
        return True
    if row[8] in bad_set:
        return True
    if row[12] in bad_set:
        return True
    return False


def det_yard_to_go(row):
    yardline_100 = int(row[8])
    posteam = row[4]
    side_of_field = row[7]
    if posteam == side_of_field:
        return min(100 - yardline_100, yardline_100)
    else:
        return max(100 - yardline_100, yardline_100)


if __name__ == '__main__':
    # train()
    check()
    # observe()
    # test()
    # plot()
