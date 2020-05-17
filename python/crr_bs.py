# Copyright © 2020 Pilara Monkgogi Moshebashebi. All rights reserved.
# This script is designed to calculate the initial price of a call option using
# the Black-Scholes and the n_th Cox-Ross-Rubinstein (CRR) formulas model.

# ------- External libraries and functions
from math import floor
from scipy.stats import binom
from numpy import linspace, exp, sqrt, log
from scipy.stats import norm
import matplotlib.pyplot as plt
# -------

# ------- n_th CRR price function


def crr_formula_call(S0, K, T, n, sigma, r):
    """
    crr_formula_call(S0, K, T, n, sigma, r) = initial price of a plain vanilla call option
    using the n_th crr model formula 

    Function inputs:
    S0 = initial asset price
    K = strike pirce
    T = expiry time
    n = number of time steps to maturity 
    sigma = volatility
    r = interest rate in the n_th crr model
    """

    # Checking input parameters
    if S0 <= 0.0 or r <= -1:
        print("Invalid input arguments")
        print("Terminating program")
        return(1)

    # Check for viability of n_th CRR model
    if n <= (1 / 4) * (sigma**2) * T:
        print("Invalid input arguments")
        print("Terminating program")
        return(1)

    # n_th CRR model parameters
    Delta_t = T / n
    R = exp(r * Delta_t)
    U = exp((r - 0.5 * (sigma**2)) * Delta_t + sigma * sqrt(Delta_t))
    D = exp((r - 0.5 * (sigma**2)) * Delta_t - sigma * sqrt(Delta_t))

    # c_th crr call price equation
    q = ((R - D) / (U - D))
    q_dash = ((R - D) / (U - D)) * (U / (R))
    A = floor((log(K / (S0 * D**n)) / log(U / D))) + 1
    price = S0 * (1 - binom.cdf(A - 1, n, q_dash)) - (K /
                                                      (R**n)) * (1 - binom.cdf(A - 1, n, q))

    return(price)

# -------

# ------- Black-Scholes price function

def black_scholes_call(S0, K, T, sigma, r):

    # Model parameters
    d_plus = (log(S0 / K) + (r + 0.5 * (sigma**2)) * T) / (sigma * sqrt(T))
    d_minus = d_plus - sigma * sqrt(T)

    # Call price equation in black-scholes equation
    price = S0 * norm.cdf(d_plus) - K * exp(-r * T) * norm.cdf(d_minus)

    return(price)
# -------

# ------- Test 0
# This section of the script is designed to test the functions crr_formula_call and
# black-shcoles_call.


# Input parameters
S0 = 100
K = 100
T = 1
n = 1
sigma = 0.15
r = 0.1

# Output
print(crr_formula_call(S0, K, T, 2, sigma, r))
# Solution: 12.20340825144055
print(black_scholes_call(S0, K, T, sigma, r))
# Solution: 11.6691284882873
# -------

# ------- Test 1
# This section of the script is designed simulate the prices of a call
# option using the above crr_formula_call function for n approaching infinity.

# Input parameters
n = list(range(0, 101, 1))
del n[0]  # We do not have a simulation when n=0.
# All other parameters remain as before.

# Initialisation
prices0 = list()

for i in n:
    prices0.append(crr_formula_call(S0, K, T, i, sigma, r))

# Plots
fig, ax = plt.subplots()
ax.plot(n, prices0, color="blue", linewidth=0.8,
        marker=".", label="n_th CRR formula ")
ax.axhline(y=11.669128, color="green", linewidth=0.8,
           label="Black-Scholes")
ax.set_xlabel("n (number of steps between t and T)")
ax.set_ylabel("call option prices")
ax.legend(fontsize="small")
plt.savefig("crr_bs0")
plt.show()
# -------

# ------- End
