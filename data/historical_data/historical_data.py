import numpy as np

n_s = 10
n_t = 12

brand_data = np.genfromtxt('diet_dataset.csv', delimiter=",")
brand_variance = 30 
brand_requirement = 10  
brand_lower_bound = np.fmax(brand_data - brand_variance, 0)  # Lower bound of brand co-branding
brand_upper_bound = brand_data + brand_variance  # Upper bound of brand co-branding
brand_probability = np.fmax(brand_data - brand_requirement, 0) / (brand_upper_bound - brand_data)  # Brand co-branding probability
brand_probability = np.fmin(brand_probability, 1)

G =brand_probability


T_offline = 50
historical_data = np.zeros((n_s, n_t))
for t in range(T_offline):
    for i in range(n_s):
        for j in range(n_t):
            feedback = np.random.binomial(1, G[i, j])
            historical_data[i, j] += feedback
historical_data /= T_offline

mu_hat = historical_data.copy()



