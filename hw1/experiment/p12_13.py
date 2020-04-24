import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

f = open("features.train", 'r')
data = f.readlines()
f.close()

def rowProcess(row):
    row = row.strip()
    row = [ x for x in row.split(' ') if x != '']
    digit, intensity, symmetry = int(float(row[0])), float(row[1]), float(row[2])
    y = 1 if digit == 8 else -1
    return [y, intensity, symmetry]

train_data = np.array([ rowProcess(row) for row in data ])
X, Y = train_data[:,1:], train_data[:,0]
C_list = [1e-5, 1e-3, 1e-1, 1e1, 1e3]
E_in, num_sv = [], []
for C in C_list:
    clf = SVC(C=C, kernel='poly', degree = 2, gamma = 1, coef0 = 1, shrinking=True, cache_size=2048)
    clf.fit(X, Y)
    error = 1 - clf.score(X, Y)
    E_in.append(error)
    num_sv.append(np.sum(np.array(clf.n_support_)))
    print(C, error, np.sum(clf.n_support_))
print(E_in)
print(num_sv)

plt.figure(1)
plt.plot([-5, -3, -1, 1, 3], E_in)
plt.xlabel('log C')
plt.ylabel('E_in')
plt.savefig('p12.png', format='png')
plt.figure(2)
plt.plot([-5, -3, -1, 1, 3], num_sv)
plt.xlabel('log C')
plt.ylabel('number of support vector')
plt.savefig('p13.png', format='png')