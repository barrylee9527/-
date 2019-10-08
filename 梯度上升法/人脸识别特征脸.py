# 特征脸
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
faces = fetch_lfw_people()  # 是一个字典的格式保存的。data,images,target,target_names,DESCR
print(faces.keys())
print(faces.data.shape)
random_indexes = np.random.permutation(len(faces.data))
x = faces.data[random_indexes]
example_faces = x[:36, :]


def plot_faces(faces):
    fig, axes = plt.subplots(6, 6, figsize=(10, 10),
                             subplot_kw={'xticks': [], 'ytick': []}, gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(faces[i].reshape(62, 47), cmap='bone')
    plt.show()


plot_faces(example_faces)
len(faces.target_names)
print(faces.target_names)

pca = PCA(svd_solver='randomized')
pca.fit(x)
print(pca.components_.shape)
plot_faces(pca.components_[:36, :])

