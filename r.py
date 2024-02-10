import numpy as np

def simplex_method(c, A, b):
    m, n = A.shape
    tableau = np.zeros((m + 1, n + m + 1))
    
    tableau[:-1, :n] = A
    tableau[:-1, n:n + m] = np.eye(m)
    tableau[:-1, -1] = b
    tableau[-1, :n] = -c
    
    while any(tableau[-1, :-1] < 0):
        pivot_col = np.argmin(tableau[-1, :-1])
        
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[tableau[:-1, pivot_col] <= 0] = np.inf  # Avoid division by zero
        
        pivot_row = np.argmin(ratios)
        
        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]
    
    return tableau[-1, -1], tableau[-1, -m-1:-1]

# Example usage:
c = np.array([4, 6])
A = np.array([[-1, -1], [1, 1],[2,5]])
b = np.array([11, 27,90])

optimal_value, optimal_solution = simplex_method(c, A, b)
print("Optimal Value:", optimal_value)
print("Optimal Solution:", optimal_solution)