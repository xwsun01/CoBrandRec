import numpy as np


def generate_synthetic_data(num_subbrands=10, num_targets=60, budget_levels=3):
    """
    Generate synthetic dataset described in the paper
    Parameters:
        num_subbrands: Number of sub-brands (default 10)
        num_targets: Number of target brands (default 60)
        budget_levels: Number of budget tiers (default 3, corresponding to low, medium, high)
    Returns:
        collaboration_probs: Collaboration probability matrix (U, V, L)
        market_gains: Market gain vector (V,)
        budget_plans: Budget allocation plans for each sub-brand (U, L)
    """
    U = num_subbrands  # Number of sub-brands
    V = num_targets    # Number of target brands
    L = budget_levels  # Number of budget tiers (low, medium, high)

    # 1. Generate compatibility parameter matrix (U, V)
    compatibility = np.random.uniform(-1, 1, size=(U, V))
    

    # 2. Define budget allocation plans for each sub-brand (low, medium, high)
    budget_plans = np.zeros((U, L))
    for u in range(U):
        max_budget = 100  # Assume the maximum budget for each sub-brand is 100
        budget_plans[u] = [
            int(max_budget * 1/3),   # Low budget
            int(max_budget * 2/3),   # Medium budget
            max_budget               # High budget
        ]
    
    # 3. Calculate collaboration probability matrix using logistic function σ(v + s)
    collaboration_probs = np.zeros((U, V, L))
    for u in range(U):
        for v in range(V):
            for l in range(L):
                s = budget_plans[u, l]
                logit = compatibility[u, v] + s
                prob = 1 / (1 + np.exp(-logit))  # Logistic function
                collaboration_probs[u, v, l] = prob

    # 4. Generate market gain vector (uniform distribution 0-1)
    market_gains = np.random.uniform(0, 1, size=V)
    
    return collaboration_probs, market_gains, budget_plans


