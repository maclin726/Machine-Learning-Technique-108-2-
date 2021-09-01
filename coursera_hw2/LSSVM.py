import numpy as np

with open('hw2_lssvm_all.dat', 'r') as f:
    raw_data =  [ line.strip().split(' ') for line in f.readlines() ]
    X_all = np.array([ list(map(float, line[:-1])) for line in raw_data ])
    y_all = np.array([ int(line[-1]) for line in raw_data ])

X_train, y_train = X_all[0:400], y_all[0:400]
X_test, y_test = X_all[400:], y_all[400:]

for g in [32, 2, 0.125]:
    for lamb in [0.001, 1, 1000]:
        K = np.array([ [ np.exp(-g * np.linalg.norm(x1-x2))  for x2 in X_train ] for x1 in X_train ])
        beta = np.dot(np.linalg.inv(lamb * np.identity(len(X_train)) + K), y_train)
        
        E_in, E_out = 0, 0
        for data, label in zip(X_train, y_train):
            predict_y = np.dot(beta.T, [ np.exp(-g * np.linalg.norm(x - data)) for x in X_train ])
            if predict_y * label < 0:
                E_in += 1
        
        for data, label in zip(X_test, y_test):
            predict_y = np.dot(beta.T, [ np.exp(-g * np.linalg.norm(x - data)) for x in X_train ])
            if predict_y * label < 0:
                E_out += 1
        
        print(f'g = {g: < 10} lambda = {lamb: < 10} E_in = {E_in / len(y_train): < 10} E_out = {E_out / len(y_test): < 10}')