import numpy as np
import matplotlib.pyplot as plt

x = np.random.random(size=100)
x = x.reshape(-1, 1)
y = x * 3. + 4. + np.random.normal(size=100)

print(x.shape)
a = np.ones((len(x), 1))
# print(a)
print(a.shape)
b = np.hstack([a, x])
x_b = np.hstack([np.ones((len(x), 1)), x])
rand_i = np.random.randint(len(x_b))
print(rand_i)
