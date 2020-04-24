import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
from sklearn.svm import SVC

f = open("features.train", 'r')
data = f.readlines()
f.close()

def rowProcess(row):
    row = row.strip()
    row = [ x for x in row.split(' ') if x != '']
    digit, intensity, symmetry = int(float(row[0])), float(row[1]), float(row[2])
    y = 1 if digit == 0 else -1
    return [y, intensity, symmetry]

data = np.array([ rowProcess(row) for row in data ])
N = len(data)
C, gamma_list = 0.1, [1e-1, 1, 1e1, 1e2, 1e3]
gamma_dict = dict({1e-1: 0, 1: 0, 1e1: 0, 1e2: 0, 1e3: 0})
for _ in tqdm(range(100)):
    valid_idx = set(np.random.choice(range(N), size = 1000).tolist())
    train_idx = set(range(N)) - valid_idx
    valid_idx, train_idx = list(valid_idx), list(train_idx)
    X_train, Y_train = data[train_idx,1:], data[train_idx,0]
    X_test, Y_test = data[valid_idx,1:], data[valid_idx,0]
    err = []
    for gamma in gamma_list:
        clf = SVC(C=C, kernel='rbf', gamma = gamma, shrinking=True, cache_size=2048)
        clf.fit(X_train, Y_train)
        err.append(1-clf.score(X_test, Y_test))
    gamma_dict[ gamma_list[err.index(min(err))] ] += 1
result = np.array([[int(math.log10(k)), v] for k, v in gamma_dict.items()])
plt.bar(result[:,0], result[:,1])
plt.title("number of times versus log(gamma)")
plt.xlabel('log(gamma)')
plt.ylabel('times')
plt.show()
