import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import pickle

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering


def k_mean_cluster(data, num_clusters: int):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(data)
    score = kmeans.score(data)
    labels = kmeans.labels_
    return score, labels


def spectral_cluster(data, num_clusters: int):
    kmeans = SpectralClustering(n_clusters=num_clusters, random_state=0).fit(data)
    labels = kmeans.labels_
    return labels


def agglomerative_cluster(data, num_clusters: int):
    agg = AgglomerativeClustering(n_clusters=num_clusters, linkage='complete').fit(data)
    return agg.labels_


def cluster_counts(labels, num_clusters: int):
    counts = np.zeros(num_clusters)
    for label in labels:
        counts[label] += 1
    print(counts)


def show_pictures(labels, choice=92):
    X = np.load('all_play_data.npy')
    num = list(labels).count(choice)
    first_dim = num // 10 + 1
    count = 0
    fig = plt.figure()
    print(num, first_dim)
    for idx, label in enumerate(labels):
        if label == choice:
            image = Image.open(X[idx, 0])
            ax = fig.add_subplot(first_dim, 10, count+1)
            ax.imshow(image, cmap='gray')
            count += 1
    plt.show()


def print_formation(labels, choice):
    data = np.load('all_play_data.npy')
    all_info = []
    for idx, label in enumerate(labels):
        if label == choice:
            game_id, play_id = det_math(data[idx, 0])
            all_info.append(float(get_info(game_id, play_id)))
    all_info.sort()
    print(np.mean(all_info))
    plt.hist(all_info, bins=20)
    plt.show()


def print_all_formations(labels):
    data = np.load('all_play_data.npy')
    plays = np.load('../../data/raw_data/numpy_data/plays.npy')
    num_labels = len(np.unique(labels))
    all_epas = [[] for _ in range(num_labels)]
    game_play_dict = {}
    for play in plays[1:, :]:
        game_id, play_id = play[0], play[1]
        epa = float(play[25])
        game_play_dict[(game_id, play_id)] = epa
    for idx, label in enumerate(labels):
        if idx % 100 == 0:
            print(idx)
        game_id, play_id = det_math(data[idx, 0])
        epa = game_play_dict[(game_id, play_id)]
        all_epas[label].append(epa)
    outfile = 'epas.p'
    pickle.dump(all_epas, open(outfile, 'wb'))


def new_team_epas(labels):
    data = np.load('all_play_data.npy')
    plays = np.load('../../data/raw_data/numpy_data/plays.npy')
    all_teams = np.unique(plays[1:, 6])
    team_mapper = {team: [] for team in all_teams}
    game_play_dict = {}
    for play in plays[1:, :]:
        game_id, play_id = play[0], play[1]
        epa = float(play[25])
        team = play[6]
        game_play_dict[(game_id, play_id)] = epa, team
    for idx, label in enumerate(labels):
        if idx % 1000 == 0:
            print(idx)
        game_id, play_id = det_math(data[idx, 0])
        epa, team = game_play_dict[(game_id, play_id)]
        team_mapper[team].append(epa)
    epa_list = [(np.median(team_mapper[team]), team) for team in team_mapper]
    epa_list.sort()
    print(epa_list)
    ys = [np.median(team_mapper[team]) for team in team_mapper]
    xs = [team_records[team] for team in team_mapper]
    plt.scatter(xs, ys)
    plt.plot(np.unique(xs), np.poly1d(np.polyfit(xs, ys, 1))(np.unique(xs)))
    plt.show()
    print(np.corrcoef(xs, ys))


def epa_plotter():
    infile = 'epas.p'
    epas = pickle.load(open(infile, "rb"))
    all_epas = [np.median(the_list) for the_list in epas]
    outfile = 'cluster_epas.p'
    pickle.dump(all_epas, open(outfile, 'wb'))


def get_info(game_id, play_id):
    plays = np.load('../../data/raw_data/numpy_data/plays.npy')
    for play in plays[1:, :]:
        if game_id == play[0] and play_id == play[1]:
            return play[25]


def cluster_chart():
    X = transform_data()
    scores = []
    clusts = [2, 3, 4, 5, 6, 8, 10, 15, 20, 30, 40, 50, 75, 100, 150,
              200, 300, 400, 500, 750, 1000]
    for i in clusts:
        score, _ = k_mean_cluster(X, i)
        scores.append(score)
        print(i, score)
    plt.plot(clusts, scores)
    plt.show()


def get_labels(filename='labels'):
    return np.load(filename)


def save_labels(labels, name='labels'):
    np.save(name, labels)


def run_pca(data):
    pca = PCA()
    pca.fit(data)
    plt.plot(pca.explained_variance_ratio_)
    plt.show()
    print(pca.singular_values_)


def transform_data():
    data = np.load('all_play_data.npy')
    scaler = preprocessing.StandardScaler().fit(data[:, 1:])
    X_scaled = scaler.transform(data[:, 1:])
    return X_scaled


team_records = {'BUF': 6.0,
                'ARI': 3.0,
                'WAS': 7.0,
                'NYJ': 4.0,
                'MIA': 7.0,
                'JAX': 5.0,
                'TEN': 9.0,
                'CLE': 7.5,
                'NYG': 5.0,
                'DEN': 6.0,
                'CIN': 6.0,
                'GB': 6.5,
                'BAL': 10.0,
                'DET': 6.0,
                'DAL': 10.0,
                'HOU': 11.0,
                'OAK': 4.0,
                'MIN': 8.5,
                'SF': 4.0,
                'SEA': 10.0,
                'CHI': 12.0,
                'CAR': 7.0,
                'PIT': 9.5,
                'PHI': 9.0,
                'NE': 11.0,
                'TB': 5.0,
                'ATL': 7.0,
                'IND': 10.0,
                'KC': 12.0,
                'LA': 13.0,
                'LAC': 12.0,
                'NO': 13.0
                }


def new_team_epas_1(labels):
    data = np.load('all_play_data.npy')
    cluster_dict = create_cluster_dict(np.unique(labels))
    all_teams = team_records.keys()
    team_mapper = {team: [] for team in all_teams}
    game_play_dict = create_game_play_dict()
    for idx, label in enumerate(labels):
        if idx % 1000 == 0:
            print(idx)
        game_id, play_id = det_math(data[idx, 0])
        epa, play_team = game_play_dict[(game_id, play_id)]
        for team in all_teams:
            #if team != play_team:
            cluster_dict[label][team].append(epa)
    for idx, label in enumerate(labels):
        if idx % 1000 == 0:
            print(idx)
        game_id, play_id = det_math(data[idx, 0])
        _, play_team = game_play_dict[(game_id, play_id)]
        print(label, play_team)
        team_mapper[play_team].append(np.median(cluster_dict[label][play_team]))
    epa_list = [(np.median(team_mapper[team]), team) for team in team_mapper]
    epa_list.sort()
    print(epa_list)
    xs = [np.median(team_mapper[team]) for team in team_mapper]
    ys = [team_records[team] for team in team_mapper]
    plt.scatter(xs, ys)
    plt.plot(np.unique(xs), np.poly1d(np.polyfit(xs, ys, 1))(np.unique(xs)))
    plt.show()
    print(np.corrcoef(xs, ys))


def create_cluster_dict(labels):
    cluster_dict = {label: {} for label in labels}
    for label in labels:
        for team in team_records:
            cluster_dict[label][team] = []
    return cluster_dict


def create_game_play_dict():
    plays = np.load('../../data/raw_data/numpy_data/plays.npy')
    game_play_dict = {}
    for play in plays[1:, :]:
        game_id, play_id = play[0], play[1]
        epa = float(play[25])
        team = play[6]
        game_play_dict[(game_id, play_id)] = epa, team
    return game_play_dict


def det_math(path: str):
    new_path = path.split('.')[0]
    game_id = new_path[17:27]
    play_id = new_path[28:]
    return game_id, play_id


def new_team_epas_2(labels):
    data = np.load('all_play_data.npy')
    plays = np.load('../../data/raw_data/numpy_data/plays.npy')
    all_teams = np.unique(plays[1:, 6])
    n_labels = len(np.unique(labels))
    team_mapper = {team: np.zeros(n_labels) for team in all_teams}
    game_play_dict = {}
    for play in plays[1:, :]:
        game_id, play_id = play[0], play[1]
        epa = float(play[25])
        team = play[6]
        game_play_dict[(game_id, play_id)] = epa, team
    for idx, label in enumerate(labels):
        # if idx % 1000 == 0:
        #     print(idx)
        game_id, play_id = det_math(data[idx, 0])
        epa, team = game_play_dict[(game_id, play_id)]
        team_mapper[team][label] += 1
    for team in team_mapper:
        team_mapper[team] /= np.sqrt(np.sum(np.square(team_mapper[team])))
    new_all_teams = [team_mapper[team] for team in team_mapper]
    new_data = np.array(new_all_teams)
    # scaler = preprocessing.StandardScaler().fit(new_data)
    # new_data = scaler.transform(new_data)
    # print(new_data.shape)
    # print(new_data)
    num_c = 4
    new_labels = agglomerative_cluster(new_data, num_c)
    # print(all_teams, new_labels)
    x = list(zip(all_teams, new_labels))
    print(x)
    # for elem in x:
    #     print(elem)
    all_clusts = {i: [] for i in range(num_c)}
    for elem in x:
        team, clust = elem
        all_clusts[clust].append(team_records[team])
    print(all_clusts)


if __name__ == "__main__":
    data = transform_data()
    score, labels = k_mean_cluster(data, 100)
    save_labels(labels, 'labels_100')
    # show_pictures(get_labels(), 92)
    # print_formation(get_labels(), 4)
    # print_all_formations(get_labels())
    # epa_plotter()
    new_team_epas_2(get_labels('labels_100.npy'))
