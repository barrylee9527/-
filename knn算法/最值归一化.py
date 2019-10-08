import numpy as np
import matplotlib.pyplot as plt

x = np.random.randint(0, 100, size=100)
ave = (x-np.min(x))/(np.max(x)-np.min(x))
print(ave)
y = np.random.randint(0, 100, (50, 2))
y = np.array(y, dtype=float)
# print(y[:10, :])
# print(np.shape(y))
y[:, 0] = (y[:, 0] - np.min(y[:, 0])) / (np.max(y[:, 0]) - np.min(y[:, 0]))
y[:, 1] = (y[:, 1] - np.min(y[:, 1])) / (np.max(y[:, 1]) - np.min(y[:, 1]))
# print(y[:10, :])
plt.scatter(y[:, 0], y[:, 1], marker='.')
plt.show()
np.mean(y[:, 0])
