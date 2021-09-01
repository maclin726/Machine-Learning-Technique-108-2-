import numpy as np

with open('hw2_adaboost_train.dat', 'r') as f:
    train_data = np.array([ list(map(float, line.split(' '))) for line in f.readlines() ])

with open('hw2_adaboost_test.dat', 'r') as f:
    test_data = np.array([ list(map(float, line.split(' '))) for line in f.readlines() ])

def learn_stump(data, weight):
    best_feature, best_threshold, best_dir, best_acc = -1, -500, 1, 0
    
    for i in range(data.shape[1]-1):
        sorted_data = data[ data[:,i].argsort() ]
        sorted_weight = weight[ data[:,i].argsort() ]
        
        threshold = [-500] \
                    + [ (sorted_data[d_idx, i] + sorted_data[d_idx+1, i]) / 2 for d_idx in range(len(data)-1) ]
        
        for dir in [1, -1]:
            for thr in threshold:
                correct = sum((sorted_data[:, -1] * (sorted_data[:, i] - thr) * dir > 0) * sorted_weight)
                wrong = sum((sorted_data[:, -1] * (sorted_data[:, i] - thr) * dir < 0) * sorted_weight)
                
                if correct > best_acc:
                    best_feature, best_threshold, best_dir, best_acc = i, thr, dir, correct
    
    return best_feature, best_threshold, best_dir


weight = np.ones(len(train_data)) / len(train_data)
stumps, stumps_alpha, epsilons = [], [], []
for t in range(300):
    stumps.append(learn_stump(train_data, weight))
    feature, threshold, dir = stumps[-1]

    stump_prediction = train_data[:,-1] * (train_data[:,feature] - threshold) * dir > 0 # a true-false array
    epsilon = sum(np.logical_not(stump_prediction) * weight) / sum(weight)
    update_factor = ((1-epsilon) / epsilon)**0.5
    epsilons.append(epsilon)

    weight = np.array([ w * update_factor if pred == False else w / update_factor for w, pred in zip(weight, stump_prediction)])
    stumps_alpha.append(np.log(update_factor))

    if t == 1:
        print(f'Problem 14: U2 is {sum(weight)}')

    if t == 299:
        print(f'Problem 15: U299 is {sum(weight)}')

print(f'Problem 16: min epsilin is {min(epsilons)}')

feature, threshold, dir = stumps[0]
g1_E_in = sum(train_data[:,-1] * (train_data[:,feature] - threshold) * dir < 0) / len(train_data)
g1_E_out = sum(test_data[:,-1] * (test_data[:,feature] - threshold) * dir < 0) / len(test_data)

print(f'Problem 12: E_in(g_1) {g1_E_in}')
print(f'Problem 17: E_out(g_1) {g1_E_out}')

train_result = np.array([ ((train_data[:,feature] - threshold) * dir > 0) for feature, threshold, dir in stumps ]) * 2 - 1
train_pred = train_result.T.dot(np.array(stumps_alpha)) * train_data[:, -1] > 0

test_result = np.array([ ((test_data[:,feature] - threshold) * dir > 0) for feature, threshold, dir in stumps ]) * 2 - 1
test_pred = test_result.T.dot(np.array(stumps_alpha)) * test_data[:, -1] > 0

print('Problem 13: E_in(G)', 1-sum(train_pred) / len(train_data))
print('Problem 18: E_out(G)', 1-sum(test_pred) / len(test_data))