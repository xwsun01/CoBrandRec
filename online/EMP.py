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
brand_lower_bound = np.fmax(brand_data - brand_variance, 0)  # Lower bound of brand co-branding
brand_upper_bound = brand_data + brand_variance  # Upper bound of brand co-branding
brand_probability = np.fmax(brand_data - brand_requirement, 0) / (brand_upper_bound - brand_data)  # Brand co-branding probability
brand_probability = np.fmin(brand_probability, 1)


# Algorithm parameters
num_sources = 10  # Number of source brands
num_targets = 12  # Number of target brands
max_selected_brands = 2  # Maximum number of brands that can be selected


# Experiment settings
num_experiments = 1  # Number of independent experiments
time_horizon = 2000  # Number of time steps
initial = 100
decay_rate = 0.995 


# Load co-branding matrix and market gain
co_branding_matrix = brand_probability
market_gain = np.genfromtxt('data/brand_data/diet/diet_market gain.csv', delimiter=",")
weighted_co_branding = co_branding_matrix * market_gain

# Initialize EMP parameters with optimistic initial values
empirical_means = np.ones((num_sources, num_targets)) * 0.8  # Optimistic initialization

# Track number of observations for each brand-target pair
observation_counts = np.ones((num_sources, num_targets))

# Track best observed combination and its revenue
best_combination = None
best_revenue = 0

# Performance metrics
revenue_over_time = np.zeros(time_horizon)
average_revenue = np.zeros(time_horizon)
cumulative_sum = np.zeros(time_horizon)

# Main experiment loop
for exp in range(num_experiments):
    np.random.seed(exp)
    start_time = timeit.default_timer()
    print('Starting experiment')

    for t in range(time_horizon):
        
        ep = max(0.1, decay_rate ** t)  
        
        if t < initial or np.random.random() < ep:
           
            selected_brands = np.random.choice(num_sources, size=max_selected_brands, replace=False)
        else:
           
            brand_revenues = np.zeros(num_sources)
            for i in range(num_sources):
                # Weight by market gain to focus on valuable targets
                brand_revenues[i] = np.sum(empirical_means[i] * market_gain)
            
            # Select top brands with highest empirical mean revenues
            selected_brands = np.argsort(brand_revenues)[-max_selected_brands:]
        
        # Observe revenues for selected brands
        for brand in selected_brands:
            # Simulate feedback from environment
            observed_revenues = (np.random.rand(num_targets) < co_branding_matrix[brand]).astype(float)
            
            # Update empirical means with smaller step size for stability
            observation_counts[brand] += 1
            learning_rate = 1 / np.sqrt(observation_counts[brand])  # Decreasing learning rate
            empirical_means[brand] = (1 - learning_rate) * empirical_means[brand] + learning_rate * observed_revenues

        # Calculate current revenue
        current_revenue = cal_rev(weighted_co_branding, selected_brands)
        
        # Update best combination if needed
        if current_revenue > best_revenue:
            best_revenue = current_revenue
            best_combination = selected_brands.copy()
        
        # Use best observed combination with some probability
        if t > initial and np.random.random() > ep and best_combination is not None:
            selected_brands = best_combination

        # Record performance metrics
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
ax.plot(average_revenue, label='EMP')
ax.legend(loc='upper left', fontsize=12)
plt.show()

# Save results
data = {
    'EMP': average_revenue
}
df = pd.DataFrame(data)
filename = 'EMP_average_revenue.csv'
df.to_csv(filename, index=False)

