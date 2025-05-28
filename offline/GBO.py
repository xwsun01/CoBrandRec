import numpy as np


# Calculate the total co-branding revenue of a given set of brands
def cal_rev(co_branding_matrix, brands):

    total_rev = np.sum((1 - np.prod(1 - co_branding_matrix[brands], axis=0)))
    return total_rev

# Greedy algorithm for budget optimization without partial enumeration
def GBO(G, B, cu, Nu, K):
    """
    Greedy algorithm for budget optimization without partial enumeration
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
    selected = np.zeros(n)  # Binary indicator of selected brands
    allocated_budget = np.zeros(n)  # Allocated budget for each brand
    remaining_budget = B
    selected_count = 0
    
    # Continue until budget is exhausted or max brands selected
    while remaining_budget > 0 and selected_count < K:
        best_brand = -1
        best_step = 0
        best_gain_ratio = 0
        
        # Find the best brand and budget allocation with highest marginal gain
        for brand in range(n):
            if selected[brand] == 0:  # Only consider unselected brands
                for step in Nu[brand]:
                    if step <= remaining_budget:  # Check if budget is sufficient
                        # Create temporary selection with this brand
                        temp_selected = np.where(selected > 0)[0].tolist()
                        temp_selected.append(brand)
                        
                        # Calculate marginal gain
                        current_revenue = cal_rev(G, np.where(selected > 0)[0])
                        new_revenue = cal_rev(G, temp_selected)
                        marginal_gain = new_revenue - current_revenue
                        
                        # Calculate gain per unit budget
                        gain_ratio = marginal_gain / step
                        
                        # Update best choice if better
                        if gain_ratio > best_gain_ratio:
                            best_gain_ratio = gain_ratio
                            best_brand = brand
                            best_step = step
        
        # If we found a valid brand to add
        if best_brand != -1:
            selected[best_brand] = 1
            allocated_budget[best_brand] = best_step
            remaining_budget -= best_step
            selected_count += 1
        else:
            # No more valid moves
            break
    
    selected_brands = np.where(selected > 0)[0]
    return selected_brands, cal_rev(G, selected_brands)

