Run splitting_segregated_parameters_non_dim.py to generate file with analytical splitting error preditions.

Run splitting_schemes_demonstration_non_dim.py with two arguments, the first for the the scheme (1-5), and the second for number of iterations (1-4). Plots of the order against Fr are saved as scheme_iterations.jpg.

Run iterate.sh to save plots for all schemes and available number of iterations.

Add "-plot_u" to plot the solutions and initial conditions for u for Fr and dt.
Add "-plot_h" to plot the solutions and initial conditions for h for Fr and dt.
Add "-print_c" to print the advective and gravitational Courant numbers.
Add "-plot_errors" to plot the predicted and measured errors for h for Fr and dt.
