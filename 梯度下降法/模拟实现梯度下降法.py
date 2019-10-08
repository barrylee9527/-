import numpy as np
import matplotlib.pyplot as plt


plot_x = np.linspace(-1, 6, 141)
plot_y = (plot_x - 2.5)**2 - 1
# plt.plot(plot_x, plot_y)
# plt.show()


def dJ(theta):
    return 2*(theta - 2.5)


def J(theta):
    try:
        return (theta-2.5)**2 - 1
    except:
        return float('inf')

epsilon = 1e-8


def gradint_descent(initial_theat, eta, n_iters=1e4, epsion=1e-8):
    theta = initial_theat
    theta_history.append(initial_theat)
    i_iter = 0
    while i_iter < n_iters:
        gradient = dJ(theta)
        last_theat = theta
        theta = theta - eta * gradient
        theta_history.append(theta)
        if abs(J(theta) - J(last_theat)) < epsilon:
            break
        i_iter += 1


""" 
theta = 0.0
eta = 0.01
epsilon = 1e-8
theta_history = [theta]
while True:
    gradient = dJ(theta)
    last_theat = theta
    theta = theta - eta * gradient
    theta_history.append(theta)
    if(abs(J(theta) - J(last_theat))<epsilon):
        break
print(len(theta_history))
plt.plot(plot_x, J(plot_x))
plt.plot(np.array(theta_history), J(np.array(theta_history)), color='r', marker='+')
plt.show()"""
eta = 1.1
theta_history = []
gradint_descent(0, eta)
print(len(theta_history))

