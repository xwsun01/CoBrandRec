import numpy as np
import matplotlib.pyplot as plt
import timeit
import pandas as pd


# Calculate the total co-branding revenue of a given set of brands
def cal_rev(co_branding_matrix, brands):

    total_rev = np.sum((1 - np.prod(1 - co_branding_matrix[brands], axis=0)))
    return total_rev


# Load co-branding data
# Use frequency to simulate probability and limit the probability value distribution to not be too sparse
brand_data = np.genfromtxt('data/brand_data/diet/diet_dataset.csv', delimiter=",")
brand_variance = 30
brand_requirement = 10
brand_lower_bound = np.fmax(brand_data - brand_variance, 0)
brand_upper_bound = brand_data + brand_variance
brand_probability = np.fmax(brand_data - brand_requirement, 0) / (brand_upper_bound - brand_data)
brand_probability = np.fmin(brand_probability, 1)


# Algorithm parameters
num_sources = 10  # Number of source brands
num_targets = 12  # Number of target brands
max_selected_brands = 3  # Maximum number of brands that can be selected


# Experiment settings
num_experiments = 1  # Number of independent experiments
time_horizon = 2000  # Number of time steps


# Initialize Thompson Sampling parameters
# Using Beta distribution Beta(β1, β2) (β1 = β2 = 1 initially) as a prior for estimating the co-branding bipartite graph model
β1 = np.ones((num_sources, num_targets))  # Success parameter, initially 1
β2 = np.ones((num_sources, num_targets))  # Failure parameter, initially 1


market_gain = np.genfromtxt('data/brand_data/diet/diet_market gain.csv', delimiter=",")
G = brand_probability * market_gain
revenue_over_time = np.zeros(time_horizon)
average_revenue = np.zeros(time_horizon)
cumulative_sum = np.zeros(time_horizon)


# Main experiment loop
for exp in range(num_experiments):
    np.random.seed(exp)
    start_time = timeit.default_timer()
    print('Starting experiment')

    for t in range(time_horizon):
        # Sample from Beta distribution for Thompson Sampling
        sampled_probabilities = np.random.beta(β1, β2)
        
        # Select top brands based on sampled probabilities
        brand_revenues = np.zeros(num_sources)
        for i in range(num_sources):
            brand_revenues[i] = np.sum(sampled_probabilities[i])
            
        # Select top brands with highest expected revenue
        selected_brands = np.argsort(brand_revenues)[-max_selected_brands:]
        
        # Get feedback for selected brands
        for brand in selected_brands:
            # Simulate feedback from environment
            feedback = (np.random.rand(num_targets) < G[brand]).astype(int)
            
            # Update Beta distribution parameters
            β1[brand] += feedback  # Increment success parameter
            β2[brand] += (1 - feedback)  # Increment failure parameter

        # Record performance metrics
        revenue_over_time[t] = cal_rev(G, selected_brands)
        if t > 0:
            cumulative_sum[t] = revenue_over_time[t] + cumulative_sum[t-1]
        else:
            cumulative_sum[t] = revenue_over_time[t]
        average_revenue[t] = (cumulative_sum[t])/(t+1)

    end_time = timeit.default_timer()
    print('Running time: ', end_time - start_time)


# Plot results
fig, ax = plt.subplots(1)
ax.plot(average_revenue, label='Thompson Sampling')
ax.legend(loc='upper left', fontsize=12)
plt.show()

# Save results
data = {
    'Thompson_Sampling': average_revenue
}
df = pd.DataFrame(data)
filename = 'Thompson_Sampling_average_revenue.csv'
df.to_csv(filename, index=False)



