import numpy as np
import matplotlib.pyplot as plt

X = [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425]
Y = [245000, 312000, 279000, 308000, 199000, 219000, 405000, 324000, 319000]

# 回归的求解就是求解回归方程的回归系数（a，b）的过程，并且使误差最小。
z1 = np.polyfit(X, Y, 1)
p1 = np.poly1d(z1)

x = np.arange(1000, 3000)
y = z1[0] * x + z1[1]
plt.figure()
plt.scatter(X, Y)
plt.plot(x, y)
plt.show()

print(p1(2000))
