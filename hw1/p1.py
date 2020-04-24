import matplotlib.pyplot as plt
import numpy as np
X = [ [1,0], [0,1], [0,-1], [-1,0], [0,2], [0,-2], [-2, 0] ]
Y = [ -1, -1, -1, 1, 1, 1, 1 ]
Z = np.array([ [x[1]**2 - 2 * x[0] - 2, x[0]**2 - 2 * x[1] - 1] for x in X ])
pos = np.array( [Z[i] for i in range(len(Z)) if Y[i] == 1] )
neg = np.array( [Z[i] for i in range(len(Z)) if Y[i] == -1] )
Pos = plt.scatter(pos[:, 0], pos[:, 1], marker = 'o', c = 'red', label = 'True Position')
Neg = plt.scatter(neg[:, 0], neg[:, 1], marker = 'o', c = 'blue')
plt.legend([Pos, Neg], ['+1', '-1'])
for z in Z:
    label = "({x}, {y})".format(x = z[0], y = z[1])
    plt.annotate(label, (z[0], z[1]), textcoords='offset points', xytext=(0,10), ha='center')
plt.xticks(np.arange(-5,5))
plt.yticks(np.arange(-5,5))
plt.title('Transformed vector')
plt.xlabel('z1')
plt.ylabel('z2')
plt.show()