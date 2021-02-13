import numpy as np
import random
import matplotlib.pyplot as plt
import PIL.Image as Image
import random
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering


class PlaybookCluster:
    def __init__(self, num_clusters, team):
        self.num_clusters = num_clusters
        self.team = team
        # self.set_seeds()
        self.cluster_dict = self.init_cluster_dict()
        self.fill_cluster_dict(team)
        self.final_clusters = None

    def init_cluster_dict(self):
        cluster_dict = []
        for i in range(self.num_clusters):
            cluster_dict.append({'count': 0,
                                 'elements': [i],
                                 'center': np.zeros(64)})
        return cluster_dict

    def set_seeds(self):
        np.random.seed(0)
        random.seed(0)

    def fill_cluster_dict(self, team: str):
        path = '../data/clustering/plays/' + team + '/clusterings.npy'
        data = np.load(path)
        for row in data:
            vec, label = row[2:-1], row[-1]
            self.cluster_dict[int(label)]['count'] += 1
            self.cluster_dict[int(label)]['center'] += vec

    def agglomerate_level(self):
        new_clusters = []
        level_keys = list(self.cluster_dict)
        random.shuffle(level_keys)
        while len(level_keys) > 1:
            chosen_index = np.random.randint(0, len(level_keys))
            chosen_cluster = level_keys.pop(chosen_index)
            other_index = np.argmin([self.find_dist(chosen_cluster, clust)
                                     for clust in level_keys])
            good_cluster = level_keys.pop(int(other_index))
            new_cluster = {'count': chosen_cluster['count'] + good_cluster['count'],
                           'elements': chosen_cluster['elements'] + good_cluster['elements'],
                           'center': chosen_cluster['center'] + good_cluster['center']}
            new_clusters.append(new_cluster)
        self.cluster_dict = new_clusters
        print(len(self.cluster_dict))

    def find_dist(self, cluster1, cluster2):
        means1 = cluster1['center']
        means2 = cluster2['center']
        return np.linalg.norm(means1 - means2)

    def agglomerate_everything(self):
        while len(self.cluster_dict) > 1:
            self.agglomerate_level()
        self.final_clusters = self.cluster_dict[0]['elements']

    def save_cluster(self, team: str):
        filename = '../data/clustering/plays/' + team + '/clusterings.npy'
        data = np.load(filename)
        first_dim = 10
        fig = plt.figure(figsize=(20, 20))
        count = 0
        for label in self.final_clusters:
            for row in data:
                if label == row[-1]:
                    pic_path = '../data/play_pics/' + team + '/class_1/' + \
                        str(int(row[0])) + '-' + str(int(row[1])) + '.png'
                    image = Image.open(pic_path)
                    ax = fig.add_subplot(first_dim, 20, count + 1)
                    ax.axis('off')
                    ax.imshow(image, cmap='gray')
                    count += 1
                    break
        path = '../agglom.png'
        plt.savefig(path)
        plt.close()
        plt.show()


if __name__ == '__main__':
    clusterer = PlaybookCluster(200, 'def')
    # clusterer.cluster_plays('def', 200)
    # clusterer.cluster_plays('off', 200)
    # clusterer.save_all_pictures('def', 200)
    # clusterer.save_all_pictures('off', 200)
    # clusterer.cluster_sizes('def')
    clusterer.agglomerate_everything()
    clusterer.save_cluster('def')
