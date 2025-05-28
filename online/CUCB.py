import numpy as np
import matplotlib.pyplot as plt
import timeit
import pandas as pd


# Calculate the total co-branding revenue of a given set of brands
def cal_rev(co_branding_matrix, brands):

    total_rev = np.sum((1 - np.prod(1 - co_branding_matrix[brands], axis=0)))
    return total_rev


# Greedy algorithm implementation for revenue maximization
def greedy(G, k_A):
    S_A = []  # Initialize selected brands set
    for _ in range(k_A):  # Loop until k_A brands are selected
        rev_out = np.zeros(num_sources)  # Initialize potential revenue array for each brand
        for i in range(num_sources):  # Iterate through all brands
            if i not in S_A:  # If brand hasn't been selected yet
                # Calculate total revenue after adding this brand
                rev_out[i] = cal_rev(G, S_A+[i])

        i_max = np.argmax(rev_out)  # Find brand with maximum potential revenue
        S_A.append(i_max)  # Add this brand to selected set
    return S_A, rev_out.max()  # Return selected brands set and maximum total revenue


# Load co-branding data
# Use frequency to simulate probability and limit the probability value distribution to not be too sparse
brand_data = np.genfromtxt('data/brand_data/diet/diet_dataset.csv', delimiter=",")
brand_variance = 30 
brand_requirement = 10  
brand_lower_bound = np.fmax(brand_data - brand_variance, 0)  # Lower bound of brand co-branding
brand_upper_bound = brand_data + brand_variance  # Upper bound of brand co-branding
brand_probability = np.fmax(brand_data - brand_requirement, 0) / (brand_upper_bound - brand_data)  # Brand co-branding probability
brand_probability = np.fmin(brand_probability, 1)


# Algorithm parameters
num_sources = 10  # Number of source brands
num_targets = 12  # Number of target brands
max_selected_brands = 4  # Maximum number of brands that can be selected


# Experiment settings
num_experiments = 1  
time_horizon = 2000 
alpha = 1.0  


# Load co-branding matrix and market gain
co_branding_matrix = brand_probability
market_gain = np.genfromtxt('data/brand_data/diet/diet_market gain.csv', delimiter=",")
weighted_co_branding = co_branding_matrix * market_gain


# Initialize CUCB parameters
mu_hat = np.ones((num_sources, num_targets)) * 0.5 
n_plays = np.ones((num_sources, num_targets))  


# Performance metrics
revenue_over_time = np.zeros(time_horizon)
average_revenue = np.zeros(time_horizon)
cumulative_sum = np.zeros(time_horizon)


# Main experiment loop
for exp in range(num_experiments):
    np.random.seed(exp)
    # Calculate initial optimal solution using greedy algorithm
    optimal_brands, optimal_revenue = greedy(weighted_co_branding, max_selected_brands)

    print('Experiment ' + str(exp) + ', Initial optimal revenue: ' + str(optimal_revenue))
    start_time = timeit.default_timer()
    print('Starting experiment')

    for t in range(time_horizon):
        # Calculate UCB values for each co-brand
        ucb_values = mu_hat + alpha * np.sqrt(np.log(t + 1) / n_plays)
        ucb_values = np.clip(ucb_values, 0, 1)  # Clip values to [0, 1]
        
        # Calculate expected revenue for each brand
        brand_revenues = np.zeros(num_sources)
        for i in range(num_sources):
            brand_revenues[i] = np.sum(ucb_values[i])
        
        current_weighted_estimates = ucb_values * market_gain

        # Call the offline oracle (greedy algorithm) with current UCB estimates
        optimal_brands, _ = greedy(current_weighted_estimates, max_selected_brands)

        # Select top brands with highest UCB values
        selected_brands = np.argsort(brand_revenues)[-max_selected_brands:]
        
        # Observe revenues for selected brands
        revenues = np.zeros((num_sources, num_targets))
        for brand in selected_brands:
           
            observed_revenues = (np.random.rand(num_targets) < co_branding_matrix[brand]).astype(float)
            revenues[brand] = observed_revenues
            
            # Update CUCB parameters
            n_plays[brand] += 1
            mu_hat[brand] = mu_hat[brand] + (observed_revenues - mu_hat[brand]) / n_plays[brand]

        revenue_over_time[t] = cal_rev(weighted_co_branding, selected_brands)
        if t > 0:
            cumulative_sum[t] = revenue_over_time[t] + cumulative_sum[t-1]
        else:
            cumulative_sum[t] = revenue_over_time[t]
        average_revenue[t] = cumulative_sum[t] / (t+1)

    end_time = timeit.default_timer()
    print(f'Running time: {end_time - start_time}')

# Plot results
fig, ax = plt.subplots(1)
ax.plot(average_revenue, label='CUCB')
ax.legend(loc='upper left', fontsize=12)
plt.show()

# Save results
data = {
    'CUCB': average_revenue
}
df = pd.DataFrame(data)
filename = 'CUCB_average_revenue.csv'
df.to_csv(filename, index=False)

