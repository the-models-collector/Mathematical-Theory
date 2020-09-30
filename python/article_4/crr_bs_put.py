# ------- Introduction
# Copyright © 2020 pilara monkgogi moshebashebi. All rights reserved.
# This script is designed to calculate the initial price of a put option in
# the Black-Scholes and the n_th Cox-Ross-Rubinstein (CRR) models.
# In the case of the CRR model we use backward induction.
# -------

# ------- External libraries and functions
from scipy.stats import binom, norm
from numpy import linspace, exp, sqrt, log
import matplotlib.pyplot as plt
# -------

# ------- n_th CRR price function


def crr_put_BI(S0, K, T, n, sigma, r):
    """
    crr_put_BI(S0, K, T, n, sigma, r) = initial price of a plain vanilla put option
    in the n_th crr model using backward induction. 

    Function inputs:
    S0 = initial asset price.
    K = strike pirce.
    T = expiry time.
    n = number of time steps to maturity. 
    sigma = volatility.
    r = interest rate in the n_th crr model.
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
    U = exp((r - 0.5 * sigma**2) * Delta_t + sigma * sqrt(Delta_t))
    D = exp((r - 0.5 * sigma**2) * Delta_t - sigma * sqrt(Delta_t))

    # Calculating terminal asset prices
    ST = list()
    for i in range(0, n + 1):
        ST.append(S0 * U ** (n - i) * D ** i)
        # Alternatively:
        # ST.append(S0 * (exp((r - 0.5 * (sigma**2)) * T) + sigma *
        #                 ((N - i) * sqrt(Delta_t) + i * (-sqrt(Delta_t)))))

    # Calculating terminal payoff of put option
    price = []
    for i in ST:
        price.append(max((K - i), 0))

    # Calculating risk neutral probabilities
    R = exp(r * Delta_t)
    q = (R - D) / (U - D)

    # Simulate time
    time = list(linspace(0, T, n + 1))
    time.pop()
    time.reverse()  # time =  (N-1)Delta_t,..., 2Delta_t, Delta_t, 0

    for i in time:
        for j in range(0, len(price) - 1):
            price[j] = ((q * price[j]) + (1 - q) *
                        price[j + 1]) / (R)
        price.pop()

    return(price[0])
# -------

# ------- Black-Scholes price function

def black_scholes_put(S0, K, T, sigma, r):

    # Model parameters
    d_plus = (log(S0 / K) + (r + 0.5 * (sigma**2)) * T) / (sigma * sqrt(T))
    d_minus = d_plus - sigma * sqrt(T)

    # Put price equation in black-scholes equation
    price = K * exp(-r * T) * norm.cdf(-d_minus) - S0 * norm.cdf(-d_plus)

    return(price)
# -------

# ------- Testing Functions
# This section of the script is designed to test the crr_put_BI function.


# Input parameters
S0 = 100
K = 100
T = 1
n = 1
sigma = 0.15
r = 0.1

# Output
print(crr_put_BI(S0, K, T, n, sigma, r))
# Solution: 2.68715005503649
print(black_scholes_put(S0, K, T, sigma, r))
# Solution: 12.203408251440539
# -------

# ------- Simulations
# This section of the script is designed simulate the prices of a put
# option using the above crr_put_BI function for n approaching infinity.

# Input parameters
n = list(range(0, 101, 1))
del n[0]  # We do not have a simulation when n=0.
# All other parameters remain as before.

# Initialisation
prices0 = list()

for i in n:
    prices0.append(crr_put_BI(S0, K, T, i, sigma, r))

# Plots
fig, ax = plt.subplots()
ax.plot(n, prices0, color="blue", linewidth=0.8,
        marker=".", label="n_th CRR put price")
ax.axhline(black_scholes_put(S0, K, T, sigma, r), color="green", linewidth=0.8,
           label="Black-Scholes put price")
ax.set_xlabel("n (number of steps between t and T)")
ax.set_ylabel("Put option prices")
ax.legend(fontsize="small")
plt.savefig("crr_bs_put")
plt.show()
# -------

# ------- End
