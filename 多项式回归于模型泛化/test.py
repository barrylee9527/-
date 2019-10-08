import numpy as np
x = np.random.uniform(-3, 3, size=100)
x = np.array(x)
# x = x.reshape(-1, 1)
print(x)
print(np.argsort(x))