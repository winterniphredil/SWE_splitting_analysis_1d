Run splitting_segregated_parameters_non_dim.py to generate file with analytical splitting error preditions.

Run splitting_schemes_demonstration_non_dim.py with choice of scheme_to_run (= 1) and iterations (= 3) to be set to run choice of scheme 1-4 with given number of iterations. The temporal order of accuracy for both u and h with respect to the Froude number, Fr, is printed and plotted. 

Change plt_u_solns = False to True (and resp. for h) to plot the solutions and initial conditions for each Fr and dt.

Change plt_errors = False to True to plot the difference between the exact and numerical solutions, and the predicted difference, in u and h for each Fr and dt.

Change print_c = False to True to print the advective and gravitational Courant numbers for each Fr and dt.

