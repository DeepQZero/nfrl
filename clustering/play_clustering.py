import numpy as np
import random
import matplotlib.pyplot as plt
import PIL.Image as Image

from sklearn import preprocessing
from sklearn.cluster import KMeans


class PlayCluster:
    def __init__(self):
        self.set_seeds()

    def set_seeds(self):
        np.random.seed(0)
        random.seed(0)

    def k_mean_cluster(self, data, num_clusters: int):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10)
        kmeans.fit(data)
        score = kmeans.score(data)
        labels = kmeans.labels_
        return score, labels

    def cluster_plays(self, team: str, num_clusters: int):
        path = '../data/autoencoder/compressions/' + team + '/compressions.npy'
        data = np.load(path)
        score, labels = self.k_mean_cluster(data[:, 2:], num_clusters)
        new_data = np.concatenate((data, labels.reshape(-1, 1)), 1)
        outpath = '../data/clustering/plays/' + team + '/clusterings.npy'
        np.save(outpath, new_data)

    def save_cluster(self, team: str, label: int):
        filename = '../data/clustering/plays/' + team + '/clusterings.npy'
        data = np.load(filename)
        num = list(data[:, -1]).count(label)
        first_dim = num // 10 + 1
        fig = plt.figure(figsize=(20, 20))
        plt.axis('off')
        count = 0
        for row in data:
            if label == row[-1]:
                pic_path = '../data/play_pics/' + team + '/class_1/' + \
                    str(int(row[0])) + '-' + str(int(row[1])) + '.png'
                image = Image.open(pic_path)
                ax = fig.add_subplot(first_dim, 10, count + 1)
                ax.axis('off')
                ax.imshow(image, cmap='gray')
                count += 1
        path = '../data/clustering/play_pics/' + \
               team + '/cluster_' + str(label)
        plt.savefig(path)
        plt.close()

    def cluster_sizes(self, team: str):
        path = '../data/clustering/plays/' + team + '/clusterings.npy'
        data = np.load(path)
        labels = data[:, -1]
        unique, counts = np.unique(labels, return_counts=True)
        print(dict(zip(unique, counts)).values())

    def transform_data(self, data):
        scaler = preprocessing.StandardScaler().fit(data[:, 2:])
        return scaler.transform(data[:, 1:])

    def save_all_pictures(self, team: str, max_label: int) -> None:
        for label in range(max_label):
            print(label)
            self.save_cluster(team, label)


if __name__ == '__main__':
    clusterer = PlayCluster()
    # clusterer.cluster_plays('def', 200)
    # clusterer.cluster_plays('off', 200)
    clusterer.save_all_pictures('def', 200)
    # clusterer.save_all_pictures('off', 200)
    # clusterer.cluster_sizes('def')
