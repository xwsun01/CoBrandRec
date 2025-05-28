import numpy as np


# Calculate the total co-branding revenue of a given set of brands
def cal_rev(co_branding_matrix, brands):

    total_rev = np.sum((1 - np.prod(1 - co_branding_matrix[brands], axis=0)))
    return total_rev


# PROP-W: Proportional budget allocation based on market gains
def PROP_W(G, B, cu, Nu, K):
    """
    PROP-W: Proportional budget allocation based on market gains
    
    Parameters:
        G: Matrix representing co-branding probabilities
        B: Total available budget for brand selection
        cu: Cost of selecting each brand
        Nu: List of possible budget allocations for each brand
        K: Maximum number of brands that can be selected
       
    Returns:
        Selected brands and their total co-branding revenue
    """
    n = len(cu)  # Number of brands
    m = G.shape[1]  # Number of sub-brands
    
    if market_gain is None:
        market_gain = np.ones(m)
    
    # Calculate the expected market gain for each brand
    expected_market_gains = np.zeros(n)
    for i in range(n):
        # Expected gain = sum of (revenue probability * market gain)
        expected_market_gains[i] = np.sum(G[i] * market_gain)
    

    # Select top K brands with highest expected market gains
    brand_scores = expected_market_gains / cu
    top_brands_indices = np.argsort(brand_scores)[-K:]
    
    # Allocate budget proportionally to the expected market gains
    selected = np.zeros(n)
    allocated_budget = np.zeros(n)
    
    # Calculate total expected market gain for selected brands
    total_expected_gain = np.sum(expected_market_gains[top_brands_indices])
    
    # Allocate budget proportionally
    remaining_budget = B
    for idx in top_brands_indices:
        # Proportion of budget based on expected market gain
        if total_expected_gain > 0:
            brand_proportion = expected_market_gains[idx] / total_expected_gain
            brand_budget = min(cu[idx], brand_proportion * B)
        else:
            brand_budget = cu[idx]  # Default to full cost if no expected gain
        
        # Find closest valid budget allocation
        valid_allocation = 0
        for step in sorted(Nu[idx]):
            if step <= brand_budget and step <= remaining_budget:
                valid_allocation = step
        
        if valid_allocation > 0:
            selected[idx] = 1
            allocated_budget[idx] = valid_allocation
            remaining_budget -= valid_allocation
    
    # Return selected brands and their total revenue
    selected_brands = np.where(selected > 0)[0]
    return selected_brands, cal_rev(G, selected_brands)


