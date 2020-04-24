import numpy as np
import matplotlib.pyplot as plt

X = [ [1,0], [0,1], [0,-1], [-1,0], [0,2], [0,-2], [-2, 0] ]
Y = [ -1, -1, -1, 1, 1, 1, 1 ]
pos = np.array( [X[i] for i in range(len(X)) if Y[i] == 1] )
neg = np.array( [X[i] for i in range(len(X)) if Y[i] == -1] )
Pos = plt.scatter(pos[:, 0], pos[:, 1], marker = 'o', c = 'red', label = 'True Position')
Neg = plt.scatter(neg[:, 0], neg[:, 1], marker = 'o', c = 'blue')
plt.legend([Pos, Neg], ['+1', '-1'])
for x in X:
    label = "({x}, {y})".format(x = x[0], y = x[1])
    plt.annotate(label, (x[0], x[1]), textcoords='offset points', xytext=(0,10), ha='center')

x = np.linspace(-2, 3, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
F1 = -4*X + 2*Y**2 - 3

alpha = [0.000,0.704,0.704,0.889,0.259,0.259,0.000]
F2 = -alpha[1] * (1+Y)**2 - alpha[2] * (1-Y)**2 + alpha[3] * (1-X)**2 + alpha[4] * (1+2*Y)**2 +  alpha[5] * (1-2*Y)**2 - 1.66

plt.contour(X, Y, F1, levels=[0], colors = 'green')
plt.contour(X, Y, F2, levels=[0], colors = 'orange')
plt.title("The vectors and curves in X-space")
plt.show()