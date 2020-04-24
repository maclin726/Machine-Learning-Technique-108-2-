import numpy as np
from sklearn import svm

X = [ [1,0], [0,1], [0,-1], [-1,0], [0,2], [0,-2], [-2, 0] ]
Y = np.array([ -1, -1, -1, 1, 1, 1, 1 ])
model = svm.SVC( C = 1e5, kernel='poly', degree=2, gamma=1, coef0=1, shrinking=False)
print(model.get_params())
model.fit(X, Y)
alpha = model.dual_coef_[0]
SVs = model.support_vectors_
print(alpha, len(alpha))
print(SVs, len(SVs))
print(model.intercept_)
svIdx = X.index(list(SVs[0]))