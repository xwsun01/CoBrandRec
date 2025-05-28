import numpy as np


# Calculate the total co-branding revenue of a given set of brands
def cal_rev(co_branding_matrix, brands):

    total_rev = np.sum((1 - np.prod(1 - co_branding_matrix[brands], axis=0)))
    return total_rev

# PROP-S: Proportional budget allocation based on the number of sub-brands
def PROP_S(G, B, cu, Nu, K):
    """
    PROP-S: Proportional budget allocation based on the number of sub-brands
    
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
    
    # Calculate the number of sub-brands each brand can influence
    sub_brand_counts = np.zeros(n)
    for i in range(n):
        # Count sub-brands with revenue probability above threshold
        threshold = 0.1  # Revenue threshold
        sub_brand_counts[i] = np.sum(G[i] > threshold)
    
    # Select top K brands with highest sub-brand counts
    # If two brands have the same count, prefer the cheaper one
    brand_scores = sub_brand_counts / cu
    top_brands_indices = np.argsort(brand_scores)[-K:]
    
    # Allocate budget proportionally to the number of sub-brands
    selected = np.zeros(n)
    allocated_budget = np.zeros(n)
    
    # Calculate total sub-brands for selected brands
    total_sub_brands = np.sum(sub_brand_counts[top_brands_indices])
    
    # Allocate budget proportionally
    remaining_budget = B
    for idx in top_brands_indices:
        # Proportion of budget based on sub-brand count
        if total_sub_brands > 0:
            brand_proportion = sub_brand_counts[idx] / total_sub_brands
            brand_budget = min(cu[idx], brand_proportion * B)
        else:
            brand_budget = cu[idx]  # Default to full cost if no sub-brands
        
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


