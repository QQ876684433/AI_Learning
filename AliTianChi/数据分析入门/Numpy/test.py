import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5, 5, .01)
y = np.arange(-5, 5, .01)

xx, yy = np.meshgrid(x, y, sparse=True)
z = np.sin(xx ** 2 + yy ** 2) / (xx ** 2 + yy ** 2)
h = plt.contourf(x, y, z)
print(h)
