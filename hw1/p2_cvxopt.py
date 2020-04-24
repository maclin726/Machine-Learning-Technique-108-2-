import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
def Kernel(x1, x2):
    return float((1+(np.dot(x2, x1)))**2)
N = 7
X = [ [1,0], [0,1], [0,-1], [-1,0], [0,2], [0,-2], [-2, 0] ]
Y = [ -1, -1, -1, 1, 1, 1, 1 ]
P = matrix(np.array([[ Y[i]*Y[j]*Kernel(X[i],X[j]) for i in range(N)] for j in range(N) ]))
q = matrix([-1. for i in range(N)])
G = matrix(-1 * np.eye(N))
h = matrix(np.zeros(N))
A = matrix([[float(Y[i])] for i in range(N)])
b = matrix(0.)
# print(P)
# print(q)
# print(G)
# print(h)
# print(A)
# print(b)
sol = solvers.qp(P, q, G, h, A, b)
print('(' + ','.join( f'{i:.3f}' for i in sol['x']) + ')')
print([sol['x'][i]*Y[i]*Kernel(1, i) for i in range(7)])
print('b = ', Y[1] - sum(sol['x'][i]*Y[i]*Kernel(X[1], X[i]) for i in range(7)))
