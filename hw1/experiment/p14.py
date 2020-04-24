import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel

f = open("features.train", 'r')
data = f.readlines()
f.close()

def rowProcess(row):
    row = row.strip()
    row = [ x for x in row.split(' ') if x != '']
    digit, intensity, symmetry = int(float(row[0])), float(row[1]), float(row[2])
    y = 1 if digit == 0 else -1
    return [y, intensity, symmetry]

train_data = np.array([ rowProcess(row) for row in data ])
X, Y = train_data[:,1:], train_data[:,0]
C_list = [1e-3, 1e-2, 1e-1, 1, 10]
K = rbf_kernel(X, gamma=80)
dist = []
for C in C_list:
    clf = SVC(C=C, kernel='rbf', gamma = 80, shrinking=True, cache_size=2048)
    clf.fit(X, Y)
    alpha_y, SVidx, svNum = np.array(clf.dual_coef_), np.array(clf.support_), len(clf.dual_coef_)
    subKmatrix = np.array([ [ K[i][j] for j in SVidx ] for i in SVidx ])
    w_len = np.sqrt(alpha_y @ subKmatrix @ alpha_y.T)[0][0]
    dist.append(1/w_len)
    print(1 / w_len)
plt.title('distance from free SVs to the hyperplane versus different C')
plt.plot([-3, -2, -1, 0, 1], dist)
plt.xlabel('log C')
plt.ylabel('distance from free SV to the hyperplane')
plt.savefig('p14.png', format='png')