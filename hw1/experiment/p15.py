import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel

f1, f2 = open("features.train", 'r'), open("features.test", "r")
train_data, test_data = f1.readlines(), f2.readlines()
f1.close()
f2.close()

def rowProcess(row):
    row = [x for x in row.strip().split(' ') if x != '']
    digit, intensity, symmetry = int(float(row[0])), float(row[1]), float(row[2])
    y = 1 if digit == 0 else -1
    return [y, intensity, symmetry]

train_data = np.array([ rowProcess(row) for row in train_data ])
test_data = np.array([ rowProcess(row) for row in test_data ])
X, Y = train_data[:,1:], train_data[:,0]
X_test, Y_test = test_data[:,1:], test_data[:,0]
gamma_list = [1, 10, 1e2, 1e3, 1e4]
C = 0.1
E_out = []
for gamma in gamma_list:
    clf = SVC(C=C, kernel='rbf', gamma = gamma, shrinking=True, cache_size=2048)
    clf.fit(X, Y)
    err = 1-clf.score(X_test, Y_test)
    E_out.append(err)
    print(err)
plt.title('E_out versus log gamma in kbf kernel SVM')
plt.plot([0, 1, 2, 3, 4], E_out)
plt.xlabel('log gamma')
plt.ylabel('E_out')
plt.savefig('p15.png', format='png')