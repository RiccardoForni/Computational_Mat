import requests
import pandas as pd
import numpy as np


key = "G2YL2E00A6PKSVYJ"



def simplex_method(c, A, b, feasible_basis, tol = 1e-4):
    m, n = A.shape
    num_iterations = 0
    

    while True:
        num_iterations += 1
        
        print('\nIteration '+str(num_iterations))

      # Get c_B, A_B and compute the current BFS and the corresponding reduced cost. 
        c_B = c[feasible_basis]
        A_B = A[:, feasible_basis]
        
        # Instead of computing inverses, we call linalg.solve from numpy to solve a linear system
        basic_feasible_solution = np.linalg.solve(A_B, b)
        reduced_costs = c - np.dot(c_B, np.linalg.solve(A_B, A))
        
        print('Current BFS: '+str(basic_feasible_solution))
        print('Reduced costs: '+str(reduced_costs))
        

        # Check for optimality
        if np.all(reduced_costs >= -tol): # Optimal solution found
            optimal_solution = np.zeros(n)
            optimal_solution[feasible_basis] = basic_feasible_solution
            optimal_value = np.dot(c, optimal_solution)
            return optimal_solution, optimal_value, feasible_basis, num_iterations

        # Choose entering variable: smallest index whose reduced cost is negative
        entering_var = min([i for i in range(n) if reduced_costs[i]<-tol]) 

        # Compute the ratios for the leaving variable
        u = np.linalg.solve(A_B, A[:,entering_var])
        if np.all(u <= 0): # if all components of u are non-negative, the problem is unbounded and we stop
            print('The problem is unbounded')
            return [], -1*np.inf, feasible_basis, num_iterations 
        
        # Select as leaving variable the index minimizing x_i/u_i, among the indices where u is positive
        nonzero_idx =  [i for i in range(len(u)) if u[i]>tol]
        leaving_var = nonzero_idx[np.argmin([basic_feasible_solution[i]/u[i] for i in nonzero_idx])]
       
        # Update basic feasible solution
        feasible_basis[leaving_var] = entering_var
        # sort the basis in increasing order, to avoid confusion with indices
        feasible_basis = sorted(feasible_basis)
        
        
def complete_to_basis(A, basis):
    m, n = A.shape
    non_basis = [i for i in range(n) if i not in basis]

    while len(basis) < m:
        for j in non_basis:
            B = A[:, basis + [j]]
            if np.linalg.matrix_rank(B) == len(basis)+1:
                basis.append(j)
                non_basis.remove(j)
                break

    return sorted(basis)
        
# Function to find a feasible basis to start the simplex method
def find_starting_vtx(A, b, tol = 1e-4):
    m, n = A.shape
    
    if np.linalg.matrix_rank(A) != m:
        print('Error: matrix A is not full row rank')
        return []
    if any(b<-tol):
        print('Error: rhs vector b must be non-negative:  ', b)
        return []
    
    #Define an auxiliary LP
    A_I =  np.concatenate((A, np.eye(m)), axis=1)
    c_aux = np.concatenate((np.zeros(n), np.ones(m)))
    
    # The following is always a feasible basis for the auxiliary LP 
    feasible_basis = [i for i in range(n, n+m)]
    
    # Run the simplex method on the auxiliary LP
    opt_sol, opt_val, feasible_basis, _ = simplex_method(c_aux, A_I, b, feasible_basis)

    # The auxiliary LP has positive optimum if and only if the original LP is infeasible
    if opt_val > tol:
        print('The LP is infeasible!')
        return []
    else:
        if max(feasible_basis)< n:
            return feasible_basis
        else:
            # This code is needed to fix the case where the feasible basis contains some index of a slack variable
            incomplete_basis = [i for i in feasible_basis if i<n]
            return complete_to_basis(A, incomplete_basis)
        


# code to solve an integer program via branch and bound. 
#int_variables contains the indices of the variables on which we can branch, i.e. the variables which are required to be integer (for instance, slack variables are not necessarily integer)
def branch_and_bound(c, A, b, int_variables=[]):
    
    if not int_variables: # all variables are set to be integers
        int_variables = [i for i in range(len(c))]
        
    num_nodes = 0

    # Initialize current best solution and value
    current_best_value = np.inf
    current_best_solution = None
    
    # Solve the LP relaxation and check that it is feasible
    optimal_solution, LP_lower_bound = solve_LP(c, A, b, print_info= False)
    if not isinstance(optimal_solution, np.ndarray):
        print('Error: the LP relaxation is infeasible!')
        return np.inf, [], num_nodes

    # The various nodes / subproblems that the B&B creates are contained in a stack, i.e. a special form of list.
    stack = [(0, optimal_solution, LP_lower_bound, c, A, b)]  # (Node ID, LP solution, LP objective value, current LP)
    
    # iterate until the stack is empty
    while stack:
      
        # get from the stack the node with best objective value (technically it's no longer a "stack")
        node_id, lp_solution, lp_obj_value, c, A, b = stack.pop(np.argmin([node[2] for node in stack]))
        print('Current node: ', node_id)
        
        if lp_obj_value >= current_best_value:
            continue  # Prune the branch if the LP objective is worse than the objective value of the current best solution

        # Check if the LP solution is integer 
        if all(np.isclose(lp_solution[int_variables], np.round(lp_solution[int_variables]), atol=1e-05)):
            # If better than the current best, update 
            if lp_obj_value < current_best_value:
                current_best_value = lp_obj_value
                current_best_solution = lp_solution
                # If current objective matches the best possible, stop
                if current_best_value == LP_lower_bound:
                    return current_best_solution, current_best_value, num_nodes
        else:
            # Choose a branching variable (heuristic: choose the ''most fractional'' variable)
            branching_variable = int_variables[np.argmax([np.abs(lp_solution[i] - np.round(lp_solution[i])) for i in int_variables])]
            branching_value = np.floor(lp_solution[branching_variable])
            
            #this is just to avoid numerical errors
            if branching_value < 0:
                branching_value = 0
    
            # Branch on the variable (add two subproblems)
            # LP 1:
            c_new, A_new, b_new = create_subproblem(c, A, b, branching_variable, branching_value)
            optimal_solution, optimal_value = solve_LP(c_new, A_new, b_new)
            
            if isinstance(optimal_solution, np.ndarray): # check if the subproblem is feasible, else we can prune the node
                num_nodes += 1    
                stack.append((num_nodes, optimal_solution, optimal_value, c_new, A_new, b_new))
            
            # LP 2:
            c_new, A_new, b_new = create_subproblem(c, A, b, branching_variable, branching_value+1, at_most= False)
            optimal_solution, optimal_value = solve_LP(c_new, A_new, b_new)
            
            if isinstance(optimal_solution, np.ndarray): # check if the subproblem is feasible, else we can prune the node
                num_nodes += 1    
                stack.append((num_nodes, optimal_solution, optimal_value, c_new, A_new, b_new))
        
        # Update LP lower bound
        LP_lower_bound = min([node[2] for node in stack]) 
            
    return current_best_solution, current_best_value, num_nodes


def solve_LP(c, A, b, print_info = False):
    # Solve the linear relaxation using simplex
    
    feasible_basis = find_starting_vtx(A, b, print_info= print_info)
    
    if feasible_basis:
        optimal_solution, optimal_value, _, _= simplex_method(c, A, b, feasible_basis, print_info = print_info)
    
        return optimal_solution, optimal_value
    else:
        return [], np.inf
    
def create_subproblem(c, A, b, idx, val, at_most=True):
    m, n = A.shape

    # Create a new column for the constraint matrix
    new_col = np.zeros((m, 1))
    A_new = np.hstack([A, new_col])

    # Create a new row for the constraint matrix
    new_row = np.zeros((1, n + 1))
    new_row[0, idx] = 1
    new_row[0, n] = 1 if at_most else -1
    A_new = np.vstack([A_new, new_row])

    # Update the right-hand side of the constraints
    b_new = np.hstack([b, val])

    # Update the objective function
    c_new = np.hstack([c, 0])

    return c_new, A_new, b_new




"""
# Example usage
c = np.array([-3, -1, 0, 0])
A = np.array([[-1, 1, 1, 0], [8, 2, 0, 1]])
b = np.array([2, 19])


integer_solution, obj_value, num_nodes = branch_and_bound(c, A, b)
print("Integer solution:", integer_solution)
print("Objective value:", obj_value)
print("B&B nodes:", num_nodes)
"""



def take_data(Symbol,time):
    df_ret = pd.DataFrame(columns = Symbol)
    df_ret.set_index("Date")
    if "Date" in Symbol:
        Symbol.remove("Date")
    for S in Symbol:
        url = "https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol="+S+"&apikey="+key+"&datatype=json"
        response = requests.get(url)
        # Parse the response and extract the data
        data = response.json()
        stock_data = data["Monthly Adjusted Time Series"]
        
        # Convert the data to a pandas DataFrame
        temp_df=pd.DataFrame(stock_data).T

        df_ret[S] = temp_df.loc[:time,"4. close"]
    return df_ret,len(Symbol),Symbol
