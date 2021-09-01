import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans

with open('hw4_nolabel_train.dat', 'r') as f:
    train_data = np.array([ list(map(float, line.strip().split(' '))) for line in f.readlines()])

inertia = []
for t in tqdm(range(100)):
    kmeans2 = KMeans(n_clusters=2).fit(train_data)
    inertia.append(kmeans2.inertia_ / len(train_data))
print('kmeans2:', sum(inertia) / len(inertia))
# kmeans2: 2.6483186289346023

inertia = []
for t in tqdm(range(100)):
    kmeans10 = KMeans(n_clusters=10).fit(train_data)
    inertia.append(kmeans10.inertia_ / len(train_data))
print('kmeans10:', sum(inertia) / len(inertia))
# kmeans10: 1.57853402240717