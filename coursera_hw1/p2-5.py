from numpy import array, dot, identity
from qpsolvers import solve_qp
import matplotlib.pyplot as plt


X = [(1, 0), (0, 1), (0, -1), (-1, 0), (0, 2), (0, -2), (-2, 0)]
Y = [-1, -1, -1, 1, 1, 1, 1]


def soft_margin(X, Y):
    P = array([[ Y[i] * Y[j] * ((1 + X[i][0]*X[j][0] + X[i][1]*X[j][1])**2) for j in range(7)] for i in range(7)]).astype(float)
    P = P + identity(7)*1e-5
    q = array([-1., -1., -1., -1., -1., -1., -1.])
    G = array([[0., 0., 0., 0., 0., 0., 0.]])
    h = array([0.])
    A = array([[-1, -1, -1,  1,  1,  1,  1]])
    b = array([0.])
    lm = array([0.] * 7)
    return solve_qp(P, q, G, h, A, b, lm)

def quadratic_transform(X, Y):
    Zx = [x[1]^2 - 2*x[0] + 3 for x in X]
    Zy = [x[0]^2 - 2*x[1] - 3 for x in X]
    colors = ['red' if y == -1 else 'blue' for y in Y]
    plt.scatter(Zx, Zy, color=colors)
    plt.grid()
    plt.savefig('p2.png')

# quadratic_transform(X, Y)   # for problem 2

alpha = soft_margin(X, Y)   # for problem 3
print('alpha of soft margin SVM is', alpha, 'sum is ', sum(alpha))

