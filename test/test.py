from BeSS import BeSS
import numpy as np

X = np.random.normal(0, 1, 100 * 10).reshape((100, 10))
y = np.random.normal(0, 1, 100)
T0 = 10
max_steps = 20
beta = np.ones(100)
weights = np.ones(100)
normal = False

a = BeSS.bess_lm(X, y, T0, max_steps, beta, weights)
b = BeSS.bess_lm(X, y, T0, max_steps, beta, weights)
print(a)





