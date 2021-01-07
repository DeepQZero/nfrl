import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

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


# play_dict['play_id'] = int(row[0])
# play_dict['game_id'] = int(row[1])
# play_dict['quarter'] = int(row[17])
# play_dict['posteam_score'] = int(row[52])
# play_dict['defteam_score'] = int(row[53])
# play_dict['down'] = int(row[18])
# play_dict['first_down_yards'] = int(row[22])
# play_dict['absolute'] = det_yard_to_go(row)
# play_dict['time_left'] = int(row[12])
# play_dict['det_reward'] = det_reward(row, final_score_dict)


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
    play_result = int(row[24])
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
        new_data = [data[0:7] + [off_label] + [def_label]]
        target = data[7]
        targets.append(target)
        datas.append(data)
X = np.array(datas)
y = targets
# print(np.unique(y))


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

log_reg = SGDClassifier(loss='log')
log_reg.fit(X, y)
for i in range(25):
    z = log_reg.predict_proba(X[i].reshape(1, -1))
    print(list(zip(z[0], np.unique(y))))


datas = []
targets = []
for key in good_data.keys():
    if key in def_labels and key in off_labels:
        data = good_data[key]
        off_label = off_labels[key]
        new_data = data[0:7]
        target = off_label
        targets.append(target)
        datas.append(data)
X = np.array(datas)
y = targets


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

log_reg = SGDClassifier(loss='log')
log_reg.fit(X, y)
for i in range(25):
    z = log_reg.predict_proba(X[i].reshape(1, -1))
    print(np.max(z))



datas = []
targets = []
for key in good_data.keys():
    if key in def_labels and key in off_labels:
        data = good_data[key]
        def_label = def_labels[key]
        new_data = data[0:7]
        target = def_label
        targets.append(target)
        datas.append(data)
X = np.array(datas)
y = targets


scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

log_reg = SGDClassifier(loss='log')
log_reg.fit(X, y)
for i in range(25):
    z = log_reg.predict_proba(X[i].reshape(1, -1))
    print(np.max(z))


# gameId, 0
# playId,
# playDescription,
# quarter,
# down,
# yardsToGo,
# possessionTeam,
# playType,
# yardlineSide,
# yardlineNumber,
# offenseFormation, 10
# personnelO,
# defendersInTheBox,
# numberOfPassRushers,
# personnelD,
# typeDropback,
# preSnapVisitorScore,
# preSnapHomeScore,
# gameClock,
# absoluteYardlineNumber,
# penaltyCodes, 20
# penaltyJerseyNumbers,
# passResult,
# offensePlayResult,
# playResult,
# epa,
# isDefensivePI