import numpy as np


# Calculate the total co-branding revenue of a given set of brands
def cal_rev(co_branding_matrix, brands):

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


