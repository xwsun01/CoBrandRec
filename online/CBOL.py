import numpy as np
import matplotlib.pyplot as plt
import timeit
import pandas as pd
import random


# Calculate the total co-branding revenue of a given set of brands
def cal_rev(co_branding_matrix, brands):
    """
    Calculate the total co-branding revenue of a given set of brands
    Parameters:
        co_branding_matrix: Matrix representing co-branding probabilities between brands
        brands: Set of selected brands
    Returns:
        Total co-branding revenue
    """
    total_rev = np.sum((1 - np.prod(1 - co_branding_matrix[brands], axis=0)))
    return total_rev


# GPE: Greedy Partial Enumeration for Budget Optimization
def Greedy_Partial_Enumeration_for_Budget_Optimization(G, B, cu, Nu, K):
    """
    GPE algorithm implementation
    Parameters:
        G: Matrix representing co-branding probabilities
        B: Total available budget for brand selection
        cu: Budget of selecting each brand 
        Nu: List of possible budget allocations for each brand
        K: Maximum number of brands that can be selected
    Returns:
        Selected brands and their total co-branding revenue
    """
    n = len(cu)
    best_allocation = np.zeros(n)
    remaining_budget = B
    possible_allocations = []
    
    # Generate all possible budget allocations using partial enumeration
    for i in range(2 ** n):
        binary_str = format(i, '0' + str(n) + 'b')
        temp_budget = np.array([int(binary_str[j]) * cu[j] for j in range(n)])
        if np.sum(temp_budget) <= B and np.sum(temp_budget > 0) <= K:
            possible_allocations.append(temp_budget)

    # Evaluate each possible budget allocation using greedy strategy
    for budget in possible_allocations:
        remaining_budget = B - np.sum(budget)
        candidate_moves = []
        
        # Generate all possible budget allocation moves
        for brand in range(n):
            for step in Nu[brand]:
                if 1 <= step <= cu[brand] - budget[brand]:
                    candidate_moves.append((brand, step))

        # Greedy selection of best budget allocation move
        while remaining_budget > 0 and candidate_moves:
            marginal_gains = []
            for (brand, step) in candidate_moves:
                temp_budget = budget.copy()
                temp_budget[brand] += step
                marginal_gain = cal_rev(G, np.where(temp_budget > 0)[0]) - \
                              cal_rev(G, np.where(budget > 0)[0])
                marginal_gains.append(marginal_gain / step)
            
            best_brand, best_step = candidate_moves[np.argmax(marginal_gains)]
            if best_step <= remaining_budget:
                budget[best_brand] += best_step
                remaining_budget -= best_step
                candidate_moves = [(n, s - best_step) if n == best_brand else (n, s) 
                                 for (n, s) in candidate_moves if s - best_step > 0]
            else:
                candidate_moves.remove((best_brand, best_step))

        # Update best budget allocation if current allocation is better
        if cal_rev(G, np.where(budget > 0)[0]) > \
           cal_rev(G, np.where(best_allocation > 0)[0]):
            best_allocation = budget.copy()

    selected_brands = np.where(best_allocation > 0)[0]
    return selected_brands, cal_rev(G, selected_brands)



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
total_budget = 10  # Total budget for brand selection
max_selected_brands = 3  # Maximum number of brands that can be selected
brand_costs = np.random.randint(1, 10, size=num_sources)  # Cost of selecting each brand
brand_requirements = [[int(brand_costs[i] / 3), int(2 * brand_costs[i] / 3), brand_costs[i]] 
                    for i in range(num_sources)]  # Possible budget allocations for each brand



# Experiment settings
num_experiments = 1  # Number of independent experiments
time_horizon = 2000  # Number of time steps
exploration_ratio = 1  # Exploration parameter for UCB



# Feedback collection parameters
feedback_counter = 0
fixed_buffer_size = 1
buffer_size_increment = 2



# Data collection
time_intervals = []
r_cbol = np.zeros((num_experiments, time_horizon))


# Load co-branding matrix and market gain
revenue_matrix = brand_probability
market_gain = np.genfromtxt('data/brand_data/diet/diet_market gain.csv', delimiter=",")
weighted_revenue = revenue_matrix * market_gain



# Load historical data for initialization
historical_data = np.genfromtxt('data/brand_data/diet/diet_historical_dataset.csv', delimiter=",")
estimated_revenue = historical_data.copy()



# Initialize CBOL parameters
estimated_market_gain = (np.random.rand(num_targets) < market_gain).astype(float)
revenue_variance = np.zeros((num_sources, num_targets))
market_gain_variance = np.zeros(num_targets)
exploration_bonus = np.zeros((num_sources, num_targets))
market_exploration_bonus = np.zeros(num_targets)
brand_visits = np.ones((num_sources, num_targets))
market_visits = np.ones(num_targets)



# Performance metrics
revenue_over_time = np.zeros(time_horizon)
average_revenue = np.zeros(time_horizon)
cumulative_sum = np.zeros(time_horizon)



# Main experiment loop
for exp in range(num_experiments):
    np.random.seed(exp)
    # Calculate optimal solution using CBOL+GPE
    optimal_brands, optimal_revenue = Greedy_Partial_Enumeration_for_Budget_Optimization(weighted_revenue, total_budget, 
                                                         brand_costs, brand_requirements, max_selected_brands)
    print('Experiment ' + str(exp) + ', Optimal revenue: ' + str(optimal_revenue))
    start_time = timeit.default_timer()
    print('Starting experiment')


    for t in range(time_horizon):
        # Calculate UCB values for CBOL
        exploration_bonus = np.sqrt(6 * revenue_variance * np.log(t + 1) / (brand_visits)) + \
                          (9 * np.log(t + 1) / (brand_visits))
        market_exploration_bonus = np.sqrt(6 * market_gain_variance * np.log(t + 1) / (market_visits)) + \
                                 (9 * np.log(t + 1) / (market_visits))
        ucb_values = np.clip(estimated_revenue + exploration_ratio * exploration_bonus, a_min=0, a_max=1)
        ucb_market = np.clip(estimated_market_gain + exploration_ratio * market_exploration_bonus, a_min=0, a_max=1)
        current_revenue = ucb_values * ucb_market

        # Select co-brands using CBOL+GPE
        if t == 0:
            selected_brands, _ = Greedy_Partial_Enumeration_for_Budget_Optimization(current_revenue, total_budget, 
                                                     brand_costs, brand_requirements, max_selected_brands)
        market_feedback = np.zeros(num_targets).astype(int)

        # Collect feedback
        for brand in selected_brands:
            brand_visits[brand] += 1
            brand_feedback = (np.random.rand(num_targets) < revenue_matrix[brand]).astype(int)
            market_feedback |= brand_feedback
            revenue_variance[brand] = (brand_visits[brand] - 1) / (brand_visits[brand] ** 2) * \
                                     ((brand_feedback - estimated_revenue[brand]) ** 2) + \
                                     (brand_visits[brand] - 1) / brand_visits[brand] * revenue_variance[brand]
            estimated_revenue[brand] = estimated_revenue[brand] + \
                                      (brand_feedback - estimated_revenue[brand]) / brand_visits[brand]
            feedback_counter += 1

            # Update brand selection based on fixed buffer size
            if feedback_counter == fixed_buffer_size:
                selected_brands, _ = Greedy_Partial_Enumeration_for_Budget_Optimization(current_revenue, total_budget, 
                                                         brand_costs, brand_requirements, max_selected_brands)
                feedback_counter = 0
                middle_time = timeit.default_timer()
                print(middle_time - start_time)
                time_intervals.append(middle_time - start_time)

        # Collect market gain feedback
        market_feedback = market_feedback.astype(bool)
        market_visits[market_feedback] += 1
        market_gain_feedback = (np.random.rand(num_targets) < market_gain).astype(int)
        market_gain_variance[market_feedback] = (market_visits[market_feedback] - 1) / \
                                              (market_visits[market_feedback] ** 2) * \
                                              ((market_gain_feedback[market_feedback] - \
                                                estimated_market_gain[market_feedback]) ** 2) + \
                                              (market_visits[market_feedback] - 1) / \
                                              market_visits[market_feedback] * market_gain_variance[market_feedback]
        estimated_market_gain[market_feedback] = estimated_market_gain[market_feedback] + \
                                               (market_gain_feedback[market_feedback] - \
                                                estimated_market_gain[market_feedback]) / \
                                               market_visits[market_feedback]

        # Record performance metrics
        revenue_over_time[t] = cal_rev(weighted_revenue, selected_brands)
        if t > 0:
            cumulative_sum[t] = revenue_over_time[t] + cumulative_sum[t-1]
        else:
            cumulative_sum[t] = revenue_over_time[t]
        average_revenue[t] = (cumulative_sum[t])/(t+1)

    r_cbol[exp] = average_revenue
    end_time = timeit.default_timer()
    print('Running time: ', end_time - start_time)



# Plot results
fig, ax = plt.subplots(1)
ax.plot(r_cbol[0], label='CBOL')
ax.legend(loc='upper left', fontsize=12)
plt.show()


# Save results
data = {
    'CBOL': r_cbol.flatten()
}
df = pd.DataFrame(data)
filename = 'average received market revenue.csv'
df.to_csv(filename, index=False)


