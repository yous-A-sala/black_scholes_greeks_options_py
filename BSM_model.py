import numpy as np
from scipy.stats import norm

## imports numpy and norm for continuous random variable

## structure gotten from derivatives market by Robert McDonald
# C = S e^(-δT) N(d_1) - K e^(-rT) N(d_2)
# S = stock price at time 0
# K = option strike price
# σ = volatility
# r = risk-free interest rate
# T = expiration time
# δ = divident yield
# N(x) = cumulative normal distribution function
# d_1 = (ln(S/K) + (r - δ + 1/2 σ^2) T) / (σ sqrt(T))
# d_2 = (ln(S/K) + (r - δ - 1/2 σ^2) T) / (σ sqrt(T)) = d_1 - σ sqrt(T)
# N(d_1) represents option delta (different form delta in function)
# N(d_2) represents risk-free probability
def black_scholes(S, K, sigma, r, T, delta, type = 'c'):
    d_1 = (np.log(S / K) + (r - delta + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d_2 = d_1 - sigma * np.sqrt(T)

    # We need to differentiate calls and puts it turns out that, for puts, we can just negate the above function
    # By the put-call parity:
    # P = C + Ke^(-rT) - Se^(-δT)
    #   = S e^(-δT) N(d_1) - K e^(-rT) N(d_2) + Ke^(-rT) - Se^(-δT)
    #   = K e^(-rT) (1 - N(d_2)) - S e^(-δT) (1 - N(d_1))
    # Note, N(-x) = P(Z <= -x) = P(Z > x) = 1 - N(x). This applies since it is a CDF.
    #   = K e^(-rT) N(-d_2) - S e^(-δT) N(-d_1)
    # thus...
    if type == 'c':
        price = S * norm.cdf(d_1) - K * np.exp(-r * T) * norm.cdf(d_2)
    elif type == 'p':
        price = K * np.exp(-r * T) * norm.cdf(-d_2) - S * norm.cdf(-d_1)
    else:
        raise ValueError("Insert c for call or p for put")
    return price

# Now to calculate greeks as a bonus
# delta = option price change when stock increases $1. Changes put/call
# gamma = change in delta as stock increases $1. Same put/call
# vega = change in option price when volatility increases 1%. Same put/call
# theta = change in option price when decrease in time to maturity by 1 day. Changes put/call
# rho = change in option price when increase in interest rate by 1%. Changes put/call
# psi = change in option price when increase in divident yield by 1%. Changes put/call
def greeks(S, K, sigma, r, T, delta, type = 'c'):
    d_1 = (np.log(S / K) + (r - delta + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d_2 = d_1 - sigma * np.sqrt(T)
    pdf_d_1 = norm.pdf(d_1) # N'(x)

    if(type == 'c'):
        delta = np.exp(-delta * T) * norm.cdf(d_1)
        theta = ((-S * np.exp(-delta * T) * pdf_d_1 * sigma) / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * norm.cdf(d_2)
        + delta * S * np.exp(-delta * T) * norm.cdf(d_1)
        rho = T * K * np.exp(-r * T) * norm.cdf(d_2)
        psi = -T * S * np.exp(-delta * T) * norm.cdf(d_1)

    elif(type == 'p'):
        delta = np.exp(-delta * T) * norm.cdf(d_1 - 1)
        theta = ((-S * np.exp(-delta * T) * pdf_d_1 * sigma) / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * norm.cdf(-d_2)
        - delta * S * np.exp(-delta * T) * norm.cdf(-d_1)
        rho = T * K * np.exp(-r * T) * norm.cdf(-d_2)
        psi = T * S * np.exp(-delta * T) * norm.cdf(d_1)
    else:
        raise ValueError("Not c or p")
    gamma = (np.exp(-delta * T) * pdf_d_1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-delta * T) * pdf_d_1 * np.sqrt(T)

    return{
        'delta' : delta,
        'gamma' : gamma,
        'theta' : theta,
        'vega' : vega,
        'rho' : rho,
        'psi' : psi
    }

# Demo using a 0 divident stock
if __name__ == "__main__":
    S = 100
    K = 100
    sigma = 0.1
    r = 0.05
    T = 1
    delta = 0

    call = black_scholes(S, K, sigma, r, T, delta, 'c')
    put = black_scholes(S, K, sigma, r, T, delta, 'p')

    call_greeks = greeks(S, K, sigma, r, T,  delta, 'c')
    put_greeks  = greeks(S, K, sigma, r, T, delta, 'p')

    print("With S = 100, K = 100, sigma = .1, r = 0.05, T = 1, and delta = 0,\n")
    print(f"Call price: {call}")
    print("Call Greeks:")
    for key, value in call_greeks.items():
        print(f"{key}: {value}")
    print(f"Put Price: {put}")
    print("Put Greeks:")
    for key, value in put_greeks.items():
        print(f"{key}: {value}")

    # User entry
    S = float(input("Stock price S: "))
    K = float(input("Strike price K: "))
    sigma = float(input("Volatility: "))
    r = float(input("Risk free interest rate: "))
    T = float(input("Time to maturity T: "))

    call = black_scholes(S, K, sigma, r, T, delta, 'c')
    put = black_scholes(S, K, sigma, r, T, delta, 'p')
    call_greeks = greeks(S, K, sigma, r, T,  delta, 'c')
    put_greeks  = greeks(S, K, sigma, r, T, delta, 'p')


    print(f"Your call price: {call}")
    for key, value in call_greeks.items():
        print(f"{key}: {value}")
    print(f"Your put Price: {put}")
    for key, value in put_greeks.items():
        print(f"{key}: {value}")