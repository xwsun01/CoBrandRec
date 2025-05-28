import numpy as np
import matplotlib.pyplot as plt
import timeit
import pandas as pd


# Calculate the total co-branding revenue of a given set of brands
def cal_rev(G, S_A):
    rev_A = np.sum((1 - np.prod(1 - G[S_A], axis=0)))
    return rev_A


# Load the brand data
# Use frequency to simulate probability and limit the probability value distribution to not be too sparse
brand_data = np.genfromtxt('data/brand_data/diet/diet_dataset.csv', delimiter=",")
brand_var = 30
brand_req = 10
brand_L = np.fmax(brand_data - brand_var, 0)
brand_U = brand_data + brand_var
brand_prob = np.fmax(brand_data - brand_req, 0) / (brand_U - brand_data)
brand_prob = np.fmin(brand_prob, 1)


# Initialize parameters
n_s = 10  # number of source brands
n_t = 12  # number of target brands
K = 2   # number of brands to select


# Experiment parameters
N_exp = 1
T = 2000
epsilon = 0.1  # exploration rate


# Initialize tracking variables
r_epsilon = np.zeros((N_exp, T))


# Load co-branding matrix
market_gain = np.genfromtxt('data/brand_data/diet/diet_market gain.csv', delimiter=",")


# Load historical data for initialization
historical_data = np.genfromtxt('data/brand_data/diet/diet_historical_dataset.csv', delimiter=",")
mu_hat = historical_data.copy()
G = brand_prob * market_gain


# Initialize visit counts
T_i = np.ones((n_s, n_t))
rev_t = np.zeros(T)
sum = np.zeros(T)
average_revenue = np.zeros(T)


for exp in range(N_exp):
    np.random.seed(exp)

    S_A_opt = np.argsort(np.sum(G, axis=1))[-K:]
    opt = cal_rev(G, S_A_opt)
    print('exp ' + str(exp) + ', opt: ' + str(opt))
    start = timeit.default_timer()
    print('Start')

    for t in range(T):
        # Epsilon-greedy strategy
        if np.random.random() < epsilon:
            # Exploration: randomly select K brands
            S_A = np.random.choice(n_s, size=K, replace=False)
        else:
            # Exploitation: select K brands with highest estimated revenue
            S_A = np.argsort(np.sum(mu_hat, axis=1))[-K:]

        # Update estimates based on feedback
        for i in S_A:
            T_i[i] += 1
            X_i = (np.random.rand(n_t) < G[i]).astype(int)
            mu_hat[i] = mu_hat[i] + (X_i - mu_hat[i]) / T_i[i]


        # Calculate average revenue
        rev_t[t] = cal_rev(G, S_A)
        if t>0:
            sum[t]=rev_t[t]+sum[t-1]
        else:
            sum[t]=rev_t[t]
        average_revenue[t]=(sum[t])/(t+1)
        

    r_epsilon[exp] = average_revenue
    stop = timeit.default_timer()
    print('Running Time: ', stop - start)


# Plot results
fig, ax = plt.subplots(1)
ax.plot(r_epsilon[0], label='Epsilon-greedy')
ax.legend(loc='upper left', fontsize=12)
plt.show()


# Save results
data = {
    'Epsilon-greedy': r_epsilon.flatten()
}
df = pd.DataFrame(data)
filename = 'epsilon_greedy_average_revenue.csv'
df.to_csv(filename, index=False)

