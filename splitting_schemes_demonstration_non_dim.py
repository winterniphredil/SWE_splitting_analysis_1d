import numpy as np
from numpy import sin, cos
from numpy.linalg import inv
import matplotlib.pyplot as plt

Fr_ops = [0.0000001, 0.0001, 0.0025, 0.01, 0.25, 0.75, 1.25] # the Froude number

scheme_to_run = 4 #CHANGE ME
iterations = 4 #CHANGE ME

L = 1
N = 20
x = np.linspace(0, L, N, endpoint=False)

k = 2*np.pi / L

u_0 = 1e-4

T = 1e-4
dx = L / N
dt_ops  = [1e-7,2e-3,5e-3,1e-2,2.5e-2,5e-2] # dt
steps = 1

h_0 = u_0
c = 1

# matrix for d/dx
off_diag = np.ones(N-1)
D = (np.diag(off_diag,1) -
     np.diag(off_diag,-1))
D[0,-1] = -1
D[-1,0] = 1
D /= (2*dx)


I_mat = np.eye(N)



def exact_update(u_0, h_0, k, Fr, c, n, dt):
    """
    Calculates the exact result after a single time step of dt for a wave mode

    Args:
        u_n (array): the velocity
        h_n (array): the height
        k (float): the wave mode
        Fr (float): the Froude number
        dt (float): the time step

    Returns:
        (tuple): the velocity, the height 
    """
    k_num = sin(k*dx)/dx
    omega_plus_num = c*(Fr+1) * k_num
    omega_minus_num = c*(Fr-1) * k_num
    exact_exp_plus = ( cos(1.*k*x) + omega_plus_num * sin(1.*k*x) * n * dt - 1/2 * (omega_plus_num)**2 * cos(1.*k*x) * n**2 * dt**2 - 1/6 * (omega_plus_num)**3 * sin(1.*k*x) * n**3 * dt**3 )
    exact_exp_minus = ( cos(1.*k*x) + omega_minus_num * sin(1.*k*x) * n * dt - 1/2 * (omega_minus_num)**2 * cos(1.*k*x) * n**2 * dt**2 - 1/6 * (omega_minus_num)**3 * sin(1.*k*x) * n**3 * dt**3 )
    return exact_exp_plus*u_0, exact_exp_plus*h_0

def scheme_1(u_n, h_n, Fr, c, dt, dx, its):
    """
    Calculates the result after a single time step of dt using 1.1

    Args:
        u_n (array): the velocity
        h_n (array): the height
        Fr (float): the Froude number
        dt (float): the time step
        dx (float): the grid spacing
        its (integer): the number of iterations

    Returns:
        (tuple): the velocity, the height 
    """
    u_i = [u_n]
    h_i = [h_n]
    A = I_mat - dt**2/4 * c * c * D @ D
    for i in range(its):
        u_prime = u_n - dt/2 * c * Fr * D @ (u_n + u_i[-1]) - dt/2 * c * D @ (h_n)
        h_rhs = h_n - dt/2 * c * Fr * D @ (h_n + h_i[-1]) - dt/2 * c * D @ (u_n + u_prime)
        h_temp = inv(A) @ h_rhs
        h_i.append(h_temp)
        u_temp = u_prime - dt/2 * c * D @ (h_i[-1])
        u_i.append(u_temp)
        #print(np.linalg.norm(u_prime) - np.linalg.norm(h_temp))
    return u_i[-1], h_i[-1]

def scheme_2_expl(u_n, h_n, Fr, c, dt, dx, its):
    """
    Calculates the result after a single time step of dt using 1.2

    Args:
        u_n (array): the velocity
        h_n (array): the height
        Fr (float): the Froude number
        dt (float): the time step
        dx (float): the grid spacing
        its (integer): the number of iterations

    Returns:
        (tuple): the velocity, the height 
    """
    u_i = [u_n]
    h_i = [h_n]
    A = I_mat - dt**2/4 * c * c * D @ D
    B = I_mat + dt/2 * c * Fr * D
    for i in range(its):
        u_rhs = u_n - dt/2 * c * Fr * D @ u_n - dt/2 * c * D @ (h_n + h_i[-1])
        u_prime = np.linalg.solve(B, u_rhs)
        u_prime_prime = u_n - dt/2 * c * Fr * D @ (u_n + u_prime) - dt/2 * c * D @ (h_n)
        h_rhs = h_n - dt/2 * c * Fr * D @ (h_n + h_i[-1]) - dt/2 * c * D @ (u_n + u_prime_prime)
        h_temp = np.linalg.solve(A, h_rhs)
        h_i.append(h_temp)
        u_i.append(u_prime_prime - dt/2 * c * D @ (h_i[-1]))
    return u_i[-1], h_i[-1]

def scheme_2(u_n, h_n, Fr, c, dt, dx, its):
    """
    Calculates the result after a single time step of dt using 1.2

    Args:
        u_n (array): the velocity
        h_n (array): the height
        Fr (float): the Froude number
        dt (float): the time step
        dx (float): the grid spacing
        its (integer): the number of iterations

    Returns:
        (tuple): the velocity, the height 
    """
    u_i = [u_n]
    h_i = [h_n]
    A = I_mat - dt**2/4 * c * c * D @ D + dt/2 * c * Fr * D
    B = I_mat + dt/2  * c * Fr * D
    for i in range(its):
        u_prime = np.linalg.solve(B, u_n - dt/2 * c * Fr * D @ (u_n) - dt/2 * c * D @ (h_n + h_i[-1]))
        u_prime_prime = u_n - dt/2 * c * Fr * D @ (u_n + u_prime) - dt/2 * c * D @ (h_n)
        h_temp = np.linalg.solve(A, h_n - dt/2 * c * Fr * D @ (h_n) - dt/2 * c * D @ (u_n + u_prime_prime))
        h_i.append(h_temp)
        u_i.append(u_prime_prime - dt/2 * c * D @ (h_i[-1]))
    return u_i[-1], h_i[-1]


def scheme_3(u_n, h_n, Fr, c, dt, dx, its):
    """
    Calculates the result after a single time step of dt using 1.3 with implicit mass advection

    Args:
        u_n (array): the velocity
        h_n (array): the height
        Fr (float): the Froude number
        dt (float): the time step
        dx (float): the grid spacing
        its (integer): the number of iterations

    Returns:
        (tuple): the velocity, the height 
    """
    u_i = [u_n]
    h_i = [h_n]
    A = I_mat - dt**2/4 * c * c * D @ D + dt/2 * c * Fr * D
    B = I_mat + dt/2 * c * Fr * D
    for i in range(its):
        u_rhs = u_n - dt/2 * c * Fr * D @ (u_n)
        u_prime = np.linalg.solve(B, u_rhs)
        h_rhs = h_n - dt/2 * c * Fr * D @ (h_n) - dt/2 * c * D @ (u_n + u_prime) + dt**2/4 * c * c * D @ D @ (h_n)
        h_temp = np.linalg.solve(A, h_rhs)
        h_i.append(h_temp)
        u_i.append(u_prime - dt/2 * c * D @ (h_n + h_i[-1]))
    return u_i[-1], h_i[-1]


def scheme_4(u_n, h_n, Fr, c, dt, dx, its):
    """
    Calculates the result after a single time step of dt using 1.3 with explicit mass advection

    Args:
        u_n (array): the velocity
        h_n (array): the height
        Fr (float): the Froude number
        dt (float): the time step
        dx (float): the grid spacing
        its (integer): the number of iterations

    Returns:
        (tuple): the velocity, the height 
    """
    u_i = [u_n]
    h_i = [h_n]
    A = I_mat - dt**2/4 * c * c * D @ D
    B = I_mat + dt/2 * c * Fr * D
    for i in range(its):
        u_rhs = u_n - dt/2 * c * Fr * D @ (u_n)
        u_prime = np.linalg.solve(B, u_rhs)
        h_rhs = h_n - dt/2 * c * Fr * D @ (h_n + h_i[-1]) - dt/2 * c * D @ (u_n + u_prime) + dt**2/4 * c * c * D @ D @ (h_n)
        h_temp = np.linalg.solve(A, h_rhs)
        h_i.append(h_temp)
        u_i.append(u_prime - dt/2 * c * D @ (h_n + h_i[-1]))
    return u_i[-1], h_i[-1]

def pred_errors(sch, its, k, Fr, c, dt):
    """
    Retrieves and calculates the predicted splitting error

    Args:
        sch (integer): the number of the scheme to retrieve
        its (integer): the number of iterations
        k (float): the wave number
        Fr (float): the Froude number
        dt (float): the time step

    Returns:
        (tuple): the velocity error, the height error
    """
    k_num = sin(k*dx)/dx
    omega_plus = c*(Fr+1)
    eval_namespace = {'Fr':Fr,'k':k_num,'c':c,'h_0':h_0,'u_0':u_0,'x':x - omega_plus*dt,'dt':dt,'cos':cos,'sin':sin}
    with open("splitting_schemes_params_non_dim_exact.txt","r") as pred_file:
        predicts = [this_pred.split("\n")[:2] for this_pred in pred_file.read().split("scheme = "+str(sch)+"\niterations = "+str(its)+"\n")][1:]
        pred_eval = []
        for i in range(len(predicts)):
            this_pred = predicts[i]
            new_pred = [(eval(j.split(" : ")[-1], eval_namespace)*dt**(i)) for j in this_pred]
            pred_eval.append(new_pred)
        u_pred = np.zeros(N)
        h_pred = np.zeros(N)
        for i in range(len(predicts)):
            u_pred += pred_eval[i][0]
            h_pred += pred_eval[i][1]
    return u_pred, h_pred

def plot_solutions_u(split, exact, orig, Fr, dt):
    """
    Plots the initial condition, analytic, and splitting scheme solution for u

    Args:
        split (array): the splitting scheme solution
        exact (array): the exact solution
        orig (array): the initial conditions
    """
    plt.plot(x,split,color='red',label='splitting scheme solution')
    plt.plot(x,exact,color='orange',label='analytic solution')
    plt.plot(x,orig,color='green',label='initial conditions')
    plt.title("Fr = "+str(Fr)+" and dt = "+str(dt))
    plt.legend()
    plt.show()
    plt.cla()

def plot_solutions_h(split, exact, orig, Fr, dt):
    """
    plots the initial condition, analytic, and splitting scheme solution for h

    Args:
        split (array): the splitting scheme solution
        exact (array): the exact solution
        orig (array): the initial conditions
    """
    plt.plot(x,split,color='red',label='splitting scheme solution')
    plt.plot(x,exact,color='orange',label='analytic solution')
    plt.plot(x,orig,color='green',label='initial conditions')
    plt.title("Fr = "+str(Fr)+" and dt = "+str(dt))
    plt.legend()
    plt.show()
    plt.cla()

def plot_errors(u_diff, h_diff, u_pred, h_pred):
    """
    Plots the numerical errors and predicted errors for both u and h

    Args:
        u_diff (array): the numerical errors for u
        h_diff (array): the numerical errors for h
        u_pred (array): the predicted errors for u
        h_pred (array): the predicted errors for h
    """
    plt.plot(x, u_diff, color='purple', label='u split difference')
    plt.plot(x, h_diff, color='gray', label='h split difference')
    plt.plot(x, u_pred, color='magenta', label='u predicted difference')
    plt.plot(x, h_pred, color='steelblue', label='h predicted difference')
    plt.title("Fr = "+str(Fr)+" and dt = "+str(dt))
    plt.legend()
    plt.show()
    plt.cla()
    
def courant_numbers(Fr, dt, dx):
    """
    Calculates and prints the advective and gravitational Courant numbers

    Args:
        sch (integer): the number of the scheme to retrieve
        its (integer): the number of iterations
        k (float): the wave number
        Fr (float): the Froude number
        dt (float): the time step

    Returns:
        (tuple): the velocity error, the height error
    """
    courant_no_adv = Fr*dt/dx
    courant_no_grv = dt/dx
    print("advection c = ",courant_no_adv)
    print("gravity c = ",courant_no_grv)

diff_errors_u = []
diff_errors_h = []
pred_errors_u = []
pred_errors_h = []



schemes = [scheme_1, scheme_2, scheme_3, scheme_4]

ratios_u = []
ratios_h = []

rat_for_fr_u = []
rat_for_fr_h = []

print("Scheme = "+str(scheme_to_run))
print("Iterations = "+str(iterations))

for i in range(len(Fr_ops)):
    Fr = Fr_ops[i]
    this_diff_u = []
    this_diff_h = []
    this_pred_u = []
    this_pred_h = []
    
    
    
    
    for j in range(len(dt_ops)):
    
        dt = dt_ops[j]
        u = u_0 * np.cos(k * x)
        h = h_0 * np.cos(k * x)
        u_n_split = u.copy()
        h_n_split = h.copy()
        u_exact, h_exact = exact_update(u_0, h_0, k, Fr, c, steps, dt)
        
        if j==0:
        
            small_dt_soln_u, small_dt_soln_h = u_n_split, h_n_split
            
            for n in range(steps):
                small_dt_soln_u, small_dt_soln_h = schemes[scheme_to_run-1](small_dt_soln_u, small_dt_soln_h, Fr, c, dt, dx, iterations)
                
            small_dt_diff_u = small_dt_soln_u - u_exact
            small_dt_diff_h = small_dt_soln_h - h_exact
            
        else:
        
            for n in range(steps):
                u_n_split, h_n_split = schemes[scheme_to_run-1](u_n_split, h_n_split, Fr, c, dt, dx, iterations)
                
            u_diff_fixed = u_n_split - u_exact - small_dt_diff_u
            h_diff_fixed = h_n_split - h_exact - small_dt_diff_h
            
            u_pred, h_pred = pred_errors(scheme_to_run, iterations, k, Fr, c, dt)
           
            this_diff_u.append(np.linalg.norm(u_diff_fixed))
            this_diff_h.append(np.linalg.norm(h_diff_fixed))
            this_pred_u.append(np.linalg.norm(u_pred))
            this_pred_h.append(np.linalg.norm(h_pred))
            
            ratios_u.append(np.linalg.norm(u_diff_fixed)/(np.linalg.norm(u_pred)))
            ratios_h.append(np.linalg.norm(h_diff_fixed)/(np.linalg.norm(h_pred)))
    
    rat_for_fr_u.append(ratios_u[5*(i)])
    rat_for_fr_h.append(ratios_h[5*(i)])
    
    diff_errors_u.append(this_diff_u)
    diff_errors_h.append(this_diff_h)
    
    grad_u_1, intercept = np.polyfit(np.log(dt_ops[1:]), np.log(this_diff_u), 1)
    print("\nFr = "+str(Fr)+"\nEstimated gradient for splitting error (u):", grad_u_1)

    grad_h_1, intercept = np.polyfit(np.log(dt_ops[1:]), np.log(this_diff_h), 1)
    print("Estimated gradient for splitting error (h):", grad_h_1)
    
    pred_errors_u.append(this_pred_u)
    pred_errors_u.append(this_pred_h)
    
    grad, intercept = np.polyfit(np.log(dt_ops[1:]), np.log(this_pred_u), 1)
    print("\nPredicted gradient for splitting error (u):", grad)

    grad, intercept = np.polyfit(np.log(dt_ops[1:]), np.log(this_pred_h), 1)
    print("Predicted gradient for splitting error (h):", grad)

grad, intercept = np.polyfit(np.log(Fr_ops), np.log(rat_for_fr_u), 1)
print("\n\nRatio gradient (using u):", grad)
grad, intercept = np.polyfit(np.log(Fr_ops), np.log(rat_for_fr_h), 1)
print("Ratio gradient (using h):", grad)

