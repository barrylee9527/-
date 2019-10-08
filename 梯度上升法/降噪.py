from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

digits = datasets.load_digits()
x = digits.data
y = digits.target
noisy_digits = x + np.random.normal(0, 4, size=x.shape)  # 这里加入随机噪声

example_digits = noisy_digits[y == 0, :][:10]
for num in range(1, 10):
    x_num = noisy_digits[y == num, :][:10]
    example_digits = np.vstack([example_digits, x_num])  # 将全部样本按照0-10的顺序叠起来，共100个样本


def plot_digits(data):  # 这个函数当做模块使用即可，不必深究
    fig, axes = plt.subplots(10, 10, figsize=(10, 10), subplot_kw={'xticks': [], 'yticks': []},
                             gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8, 8), cmap='binary', interpolation='nearest', clim=(0, 16))
    plt.show()


plot_digits(example_digits)

pca = PCA(0.5)  # 因为本身噪声比较大，所以这里仅保留原始数据0.5的信息
pca.fit(noisy_digits)
print(pca.n_components_)  # 使用了12个特征

components = pca.transform(example_digits)
filtered_digits = pca.inverse_transform(components)
plot_digits(filtered_digits)