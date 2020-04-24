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
    y = 1 if digit == 0 else -1
    return [y, intensity, symmetry]

train_data = np.array([ rowProcess(row) for row in data ])
X, Y = train_data[:,1:], train_data[:,0]
C_list = [1e-5, 1e-3, 1e-1, 1, 3]
lens = []
for C in C_list:
    clf = SVC(C=C, kernel='linear', shrinking=True, cache_size=2000)
    clf.fit(X, Y)
    w, b = clf.coef_, clf.intercept_
    length = (np.linalg.norm(np.array(w))**2 + b**2)**0.5
    lens.append(length)
    print(C, w, b, length)
plt.title('length of w versus different C')
plt.plot([-5, -3, -1, 1, 3], lens)
plt.xlabel('log C')
plt.ylabel('length of w')
plt.savefig('p11.png', format='png')