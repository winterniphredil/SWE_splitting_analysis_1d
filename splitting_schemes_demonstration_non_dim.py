import numpy as np
from numpy import sin, cos
import scipy.linalg
from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt
from operator import add

m_ops = [1,2,3,4,5] # the wavenumber 
Fr_ops = [0.05, 0.1, 0.2, 0.5, 0.75] # the Froude number

scheme_to_run = 1 #CHANGE ME
iterations = 3 #CHANGE ME

L = 1e-3
N = 256
x = np.linspace(0, L, N, endpoint=False)


u_0 = 1e-4

T = 1e-5
dx = L / N
dt_ops  = [1e-7,5e-7,1e-6,2.5e-6,5e-6] # dt
steps = 1


# matrix for d/dx
off_diag = np.ones(N-1)
D = (np.diag(off_diag,1) -
     np.diag(off_diag,-1))
D[0,-1] = -1
D[-1,0] = 1
D /= (2*dx)
# matrix for d^2/dx^2
diag = - 2 * np.ones(N)
D2 = (np.diag(diag) +
      np.diag(off_diag,1) +
      np.diag(off_diag,-1))
D2[0,-1] = 1
D2[-1,0] = 1
D2 /= dx**2

I_mat = np.eye(N)



def d_dx(fn):
    """
    Applies the first spatial differential operator, D

    Args:
        fn (array): the array in x to be differentiated

    Returns:
        (array): the values of the operator applied to fn
    """
    return D @ fn
    
def d2_dx2(fn):
    """
    Applies the second spatial differential operator, D2

    Args:
        fn (array): the array in x to be differentiated

    Returns:
        (array): the values of the operator applied to fn
    """
    return D2 @ fn

def exact_update_no_exp(u_n, h_n, k, Fr, dt):
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
    return u_n/2*(np.cos(k*(1+1/Fr)*dt)+np.cos(k*(1-1/Fr)*dt))+h_n/2*(np.sin(k*(1+1/Fr)*dt)+np.sin(k*(1-1/Fr)*dt)), h_n/2*(np.cos(k*(1+1/Fr)*dt)+np.cos(k*(1-1/Fr)*dt))+u_n/2*(np.sin(k*(1+1/Fr)*dt)+np.sin(k*(1-1/Fr)*dt))

def scheme_1(u_n, h_n, Fr, dt, dx, its):
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
    A = I_mat - dt**2/4 * 1/Fr**2 * D2
    for i in range(its):
        u_prime = u_n - dt/2 * d_dx(u_n + u_i[-1]) - dt/2 * 1/Fr**2 * d_dx(h_n)
        h_temp = np.linalg.solve(A, h_n - dt/2 * d_dx(h_n + h_i[-1]) - dt/2 * d_dx(u_n + u_prime))
        h_i.append(h_temp)
        u_i.append(u_prime - dt/2 * 1/Fr**2 * d_dx(h_i[-1]))
    return u_i[-1], h_i[-1]

def scheme_2(u_n, h_n, Fr, dt, dx, its):
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
    A = I_mat - dt**2/4 * 1/Fr**2 * D2
    B = I_mat + dt/2  * D
    for i in range(its):
        u_prime = np.linalg.solve(B, u_n - dt/2 * d_dx(u_n) - dt/2 * 1/Fr**2 * d_dx(h_n + h_i[-1]))
        u_prime_prime = u_n - dt/2 * d_dx(u_n + u_prime) - dt/2 * 1/Fr**2 * d_dx(h_n)
        h_temp = np.linalg.solve(A, h_n - dt/2 * d_dx(h_n + h_i[-1]) - dt/2 * d_dx(u_n + u_prime_prime))
        h_i.append(h_temp)
        u_i.append(u_prime_prime - dt/2 * 1/Fr**2 * d_dx(h_i[-1]))
    return u_i[-1], h_i[-1]

def scheme_2_orig(u_n, h_n, Fr, dt, dx, its):
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
    A = I_mat - dt**2/4 * 1/Fr**2 * D2 + dt/2 * D
    B = I_mat + dt/2  * D
    for i in range(its):
        u_prime = np.linalg.solve(B, u_n - dt/2 * d_dx(u_n) - dt/2 * 1/Fr**2 * d_dx(h_n + h_i[-1]))
        u_prime_prime = u_n - dt/2 * d_dx(u_n + u_prime) - dt/2 * 1/Fr**2 * d_dx(h_n)
        h_temp = np.linalg.solve(A, h_n - dt/2 * d_dx(h_n) - dt/2 * d_dx(u_n + u_prime_prime))
        h_i.append(h_temp)
        u_i.append(u_prime_prime - dt/2 * 1/Fr**2 * d_dx(h_i[-1]))
    return u_i[-1], h_i[-1]


def scheme_3(u_n, h_n, Fr, dt, dx, its):
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
    A = I_mat - dt**2/4 * 1/Fr**2 * D2 + dt/2 * D
    B = I_mat + dt/2  * D
    for i in range(its):
        u_prime = np.linalg.solve(B, u_n - dt/2 * d_dx(u_n))
        h_temp = np.linalg.solve(A, h_n - dt/2 * d_dx(h_n) - dt/2 * d_dx(u_n + u_prime) + dt**2/4 * 1/Fr**2 * d2_dx2(h_n))
        h_i.append(h_temp)
        u_i.append(u_prime - dt/2 * 1/Fr**2 * d_dx(h_n + h_i[-1]))
    return u_i[-1], h_i[-1]


def scheme_4(u_n, h_n, Fr, dt, dx, its):
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
    A = I_mat - dt**2/4 * 1/Fr**2 * D2
    B = I_mat + dt/2 * D
    for i in range(its):
        u_prime = np.linalg.solve(B, u_n - dt/2 * d_dx(u_n))
        h_temp = np.linalg.solve(A, h_n - dt/2 * d_dx(h_n + h_i[-1]) - dt/2 * d_dx(u_n + u_prime) + dt**2/4 * 1/Fr**2 * d2_dx2(h_n))
        h_i.append(h_temp)
        u_i.append(u_prime - dt/2 * 1/Fr**2 * d_dx(h_n + h_i[-1]))
    return u_i[-1], h_i[-1]

def pred_errors(sch, its, k, Fr, dt):
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
    with open("splitting_schemes_params_non_dim_exact.txt","r") as pred_file:
        predicts = [this_pred.split("\n")[:2] for this_pred in pred_file.read().split("scheme = "+str(sch)+"\niterations = "+str(its)+"\n")][2:]
        
        for i in range(len(predicts)):
            this_pred = predicts[i]
            new_pred = [(eval(j.split(" : ")[-1])*dt**i) for j in this_pred]
            predicts[i] = new_pred
        
        u_pred = [0]*N
        h_pred = [0]*N
        for i in range(len(predicts)):
            if not isinstance(predicts[i][0],float): u_pred = list(map(add, predicts[i][0], u_pred))
            if not isinstance(predicts[i][1],float): h_pred = list(map(add, predicts[i][1], h_pred))
    return np.linalg.norm(u_pred), np.linalg.norm(h_pred)



diff_errors_u = []
diff_errors_h = []
pred_errors_u = []
pred_errors_h = []

k_ops = np.array([2*np.pi * m / L for m in m_ops])

k = 2*np.pi / L

schemes = [scheme_1, scheme_2, scheme_3, scheme_4]

print("Scheme = "+str(scheme_to_run))
print("Iterations = "+str(iterations))

for Fr in Fr_ops:
    this_diff_u = []
    this_diff_h = []
    this_pred_u = []
    this_pred_h = []
    
    h_0 = Fr * u_0
    
    
    for dt in dt_ops:
    #for m in m_ops:
    #    k = 2*np.pi * m / L
        u = u_0 * np.cos(k * x)
        h = h_0 * np.cos(k * x)
        u_n_split = u.copy()
        h_n_split = h.copy()
        u_exact = u.copy()
        h_exact = h.copy()
        
        #steps = int(T/dt)
        
        for _ in range(steps):
            u_n_split, h_n_split = schemes[scheme_to_run-1](u_n_split, h_n_split, Fr, dt, dx, iterations)
            u_exact, h_exact = exact_update_no_exp(u_exact, h_exact, k, Fr, dt)
        
        plt.plot(x,u_n_split,color='red')
        plt.plot(x,u_exact,color='orange')
        plt.plot(x,h_n_split,color='blue')
        plt.plot(x,h_exact,color='green')
        plt.title("Fr = "+str(Fr)+" and dt = "+str(dt))
        plt.pause(0.5)
        plt.cla()
        
        u_pred, h_pred = pred_errors(scheme_to_run, iterations, k, Fr, dt)
        
        this_diff_u.append(np.linalg.norm(u_n_split - u_exact))
        this_diff_h.append(np.linalg.norm(h_n_split - h_exact))
        this_pred_u.append(u_pred)
        this_pred_h.append(h_pred)
    
    #plt.loglog(k_ops, this_diff_u, 'o-')
    #plt.show()
    #k_ops = np.array(k_ops)
    
    diff_errors_u.append(this_diff_u)
    diff_errors_h.append(this_diff_h)
    
    grad, intercept = np.polyfit(np.log(dt_ops), np.log(this_diff_u), 1)
    print("\nFr = "+str(Fr)+"\nEstimated gradient for splitting error (u):", grad)

    grad, intercept = np.polyfit(np.log(dt_ops), np.log(this_diff_h), 1)
    print("Estimated gradient for splitting error (h):", grad)
    
    pred_errors_u.append(this_pred_u)
    pred_errors_u.append(this_pred_h)
    
    grad, intercept = np.polyfit(np.log(dt_ops), np.log(this_pred_u), 1)
    print("\nPredicted gradient for splitting error (u):", grad)

    grad, intercept = np.polyfit(np.log(dt_ops), np.log(this_pred_h), 1)
    print("Predicted gradient for splitting error (h):", grad)


#print(diff_errors_u)
#print(diff_errors_h)
#print(pred_errors_u)
#print(pred_errors_h)
