import numpy as np
import graphviz
import random
from tqdm import tqdm
from sklearn import tree
from sklearn import ensemble

with open('hw3_dectree_train.dat', 'r') as f:
    train_data = np.array([list(map(float, line.strip().split(' '))) for line in f.readlines()])
    train_X, train_Y = train_data[:,:-1], train_data[:, -1]
    train_Y[train_Y == -1] = 0
    N = len(train_X)

with open('hw3_dectree_test.dat', 'r') as f:
    test_data = np.array([list(map(float, line.strip().split(' '))) for line in f.readlines()])
    test_X, test_Y = test_data[:,:-1], test_data[:, -1]
    test_Y[test_Y == -1] = 0



clf = tree.DecisionTreeClassifier(criterion='gini',
                                  min_samples_split=2,
                                  min_samples_leaf=1)
clf.fit(train_X, train_Y)

# Problem 13, 14, 15
dot_data = tree.export_graphviz(clf)
graph = graphviz.Source(dot_data, format='png')
graph.render('tree')

print(f'Problem 14: E_in is {1 - clf.score(train_X, train_Y)}')
print(f'Problem 15: E_out is {1 - clf.score(test_X, test_Y)}')

# Problem 16, 17, 18
# clf = ensemble.RandomForestClassifier(n_estimators=300,
#                                      criterion='gini',
#                                      min_samples_split=2,
#                                      min_samples_leaf=1,
#                                      max_features=None)

# tree_E_in = []
# E_in, E_out = [], []
# times = 100
# for t in tqdm(range(times)):
#     clf.fit(train_X, train_Y)
#     E_in.append(1-clf.score(train_X, train_Y))
#     E_out.append(1-clf.score(test_X, test_Y))

#     for tree in clf.estimators_:
#         tree_E_in.append(1-tree.score(train_X, train_Y))

# print(sum(tree_E_in)/(300*times))
# print(sum(E_in)/times)
# print(sum(E_out)/times)


# Problem 19, 20
clf = ensemble.RandomForestClassifier(n_estimators=300,
                                     criterion='gini',
                                     max_depth=1,
                                     max_features=None)
E_in, E_out = [], []
times = 100
for t in tqdm(range(times)):
    clf.fit(train_X, train_Y)
    E_in.append(1-clf.score(train_X, train_Y))
    E_out.append(1-clf.score(test_X, test_Y))

print(sum(E_in)/times)
print(sum(E_out)/times)
