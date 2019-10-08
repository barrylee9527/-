import numpy as np
import matplotlib.pyplot as plt
from 机器学习.梯度上升法.PCA import PCA

x = np.empty((100, 2))
x[:, 0] = np.random.uniform(0., 100., size=100)
x[:, 1] = 0.75 * x[:, 0] + 3. + np.random.normal(0, 10., size=100)
pca = PCA(n_components=2)
pca.fit(x)
print(pca.components)
pca = PCA(1)
pca.fit(x)
x_reduction = pca.transform(x)
print(x_reduction.shape)
x_restore = pca.inverse_transform(x_reduction)
print(x_restore.shape)
plt.scatter(x[:, 0],x[:, 1], color='b', alpha=0.5)
plt.scatter(x_restore[:, 0], x_restore[:, 1], color='r', alpha=0.5)
plt.show()
