import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
x = np.empty((100, 2))
x[:, 0] = np.random.uniform(0., 100., size=100)
x[:, 1] = 0.75 * x[:, 0] + 3 + np.random.normal(0, 5, size=100)
plt.scatter(x[:, 0], x[:, 1])
plt.show()
pca = PCA(n_components=1)
pca.fit(x)
x_reduction = pca.transform(x)
x_restore = pca.inverse_transform(x_reduction)
plt.scatter(x_restore[:, 0], x_restore[:, 1])
plt.show()
# 数字识别
from sklearn import datasets
digits = datasets.load_digits()
x = digits.data
y = digits.target
noisy_digits = x + np.random.normal(0, 4, size=x.shape)
example_digits = noisy_digits[y == 0, :][:10]
for num in range(1, 10):
    x_num = noisy_digits[y == num, :][:10]
    example_digits = np.vstack([example_digits, x_num])


def plot_digits(data):
    fig, axes = plt.subplots(10, 10, figsize=(10, 10),
                             subplot_kw={'xticks': [], 'ytick': []}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8), cmap='binary', interpolation='nearest', clim=(0, 16))
    plt.show()


plot_digits(example_digits)
pca = PCA(0.5)
pca.fit(noisy_digits)
print(pca.components_)
components = pca.transform(example_digits)
filtered_digits = pca.inverse_transform(components)
plot_digits(filtered_digits)

