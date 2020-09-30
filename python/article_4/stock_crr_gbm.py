# ------- Introduction
# Copyright © 2020 Pilara Monkgogi Moshebashebi. All rights reserved.
# This script is designed to simulate the sample path of a stock in
# the Black-Scholes and n_th CRR models.
# -------

# ------- External libraries and functions
import numpy
import random
# -------

# ------- Black-Scholes Model

# Model parameters
S0 = 10  # initial stock price
r = 0.15  # interest rate
sigma = 0.1  # volatility

# Time parameters
t = 0  # start time
T = 10  # end time

# Number of time steps
n0 = 1000  # number of time steps between t and T (in Black-Scholes Model)

# Other vectors and variables
W = 0  # brownian motion variable initialised to zero (i.e. W_0=0)
BS_price_v = [S0]  #  stock prices vector with BS_Price[0] = 0

for i in numpy.linspace(0, T, n0 + 1):
    if i == 0:
        continue

    W = W + ((T / n0) ** 0.5) * numpy.random.standard_normal()
    BS_price_v.append(S0 * numpy.exp((r - 0.5 * sigma**2) * i + sigma * W))

# -------

# ------- n_th CRR Model

# Time parameters
n1 = 4  # number of time steps between t and T (in CRR Model)
delta_t = T / n1  # change in time
step_size = delta_t**0.5  # step size

# Other vectors and variables
rw = 0  # random walk variable initialised to zero
CRR_price_v = [S0]  # stock price vector with CRR_price[0] = S0

for i in numpy.linspace(0, T, n1 + 1):
    if i == 0:
        continue

    z = random.choice([1, -1])
    rw = rw + step_size * z
    price = S0 * numpy.exp((r - 0.5 * sigma**2) * i +
                           sigma * rw)
    CRR_price_v.append(price)
# -------

# # ------- Plots
# # print(rw_vector)
import matplotlib.pyplot as plt
plt.plot(list(numpy.linspace(0, T, n0 + 1)),
         BS_price_v,  color="blue", linewidth=0.8, label="BS model price")
plt.plot(list(numpy.linspace(0, T, n1 + 1)),
         CRR_price_v, color="green", linewidth=0.8, label="Approximate equation price")
plt.scatter(list(numpy.linspace(0, T, n1 + 1)),
            CRR_price_v, color="red", s=10, label="n_th CRR model price")
plt.axvline(x=0, linewidth=0.8, color="black", linestyle='dashed')
plt.axhline(y=10, linewidth=0.8, color="black", linestyle='dashed')
plt.xlabel("time")
plt.ylabel("stock price")
plt.legend(fontsize="small")
plt.savefig("figure_0")
plt.show()
# -------

# ------- end
