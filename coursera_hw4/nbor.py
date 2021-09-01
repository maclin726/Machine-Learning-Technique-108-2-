import numpy as np
from sklearn.neighbors import KNeighborsClassifier

with open('hw4_nbor_train.dat', 'r') as f:
    train_data = np.array([ list(map(float, line.strip().split(' '))) for line in f.readlines()])
    train_X, train_y = train_data[:, :-1], train_data[:,-1]

with open('hw4_nbor_test.dat', 'r') as f:
    test_data = np.array([ list(map(float, line.strip().split(' '))) for line in f.readlines()])
    test_X, test_y = test_data[:, :-1], test_data[:, -1]

neigh = KNeighborsClassifier(n_neighbors=1, algorithm='brute')
neigh.fit(train_X, train_y)
print(f'E_in of 1-neighbor: {1-neigh.score(train_X, train_y)}')
print(f'E_out of 1-neighbor: {1-neigh.score(test_X, test_y)}')

neigh = KNeighborsClassifier(n_neighbors=5, algorithm='brute')
neigh.fit(train_X, train_y)
print(f'E_in of 5-neighbor: {1-neigh.score(train_X, train_y)}')
print(f'E_out of 5-neighbor: {1-neigh.score(test_X, test_y)}')

# E_in of 1-neighbor: 0.0
# E_out of 1-neighbor: 0.344
# E_in of 5-neighbor: 0.16000000000000003
# E_out of 5-neighbor: 0.31599999999999995