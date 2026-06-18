import sympy as smp
from sympy import sin, symbols
import numpy as np
import matplotlib.pyplot as plt

k, Fr, c, c_a, c_g = symbols('k Fr c c_a c_g', real = True)
dt, dx, theta = symbols('dt dx theta', real = True)
theta = symbols('theta', real=True)
substitutions = {Fr: c_a/c_g, k: 1j*sin(theta)/dx, dt: c_g*dx/c}

id_mat = smp.Matrix([[1,0],[0,1]])

its = 2


def semi_implicit(ampl_mat_prev_iter):
    prev_iter_contr = smp.Matrix([[0,0],[-c*dt/2*k,-c*Fr*dt/2*k]]) * ampl_mat_prev_iter
    bracket_contr = smp.Matrix([[1,0],[-c*dt/2*k,-c*Fr*dt/2*k]]) * ( smp.Matrix([[1-c*Fr*dt/2*k,-c*dt/2*k],[0,1]]) + smp.Matrix([[-c*Fr*dt/2*k,0],[0,0]]) * ampl_mat_prev_iter )
    ampl_mat = smp.Matrix([[1,-c*dt/2*k],[0,1]]) * smp.Matrix([[1,0],[0,1-c*c*dt*dt/4*k*k]]).inv() * (prev_iter_contr + bracket_contr)
    return ampl_mat
    
def segregated(ampl_mat_prev_iter):
    bracket_1 = smp.Matrix([[1-c*Fr*dt/2*k,-c*dt/2*k],[0,1]]) + smp.Matrix([[0,-c*dt/2*k],[0,0]]) * ampl_mat_prev_iter
    bracket_coeff_1 = smp.Matrix([[0,0],[c*c*Fr*dt*dt/4*k*k,0]]) * smp.Matrix([[1+c*Fr*dt/2*k,-0],[0,1]]).inv()
    term_1 = smp.Matrix([[1,0],[c*c*Fr*dt*dt/4*k*k - c*dt/2*k , 1 - c*Fr*dt/2*k + c*c*dt*dt/4*k*k]])
    bracket_coeff_2 = smp.Matrix([[1, -c*Fr*dt/2*k],[0,1]]) * smp.Matrix([[1,0],[0, 1 + c*Fr*dt/2*k - c*c*dt*dt/4*k*k]]).inv()
    ampl_mat = bracket_coeff_2 * (term_1 + bracket_coeff_1 * bracket_1)
    return ampl_mat

def op_split_1(ampl_mat_prev_iter):
    term_1 = smp.Matrix([[0,0],[-c/2*dt*k,0]]) * smp.Matrix([[1+c*Fr/2*dt*k,0],[0,1]]).inv() * smp.Matrix([[1-c*Fr/2*dt*k,0],[0,1]])
    term_2 = smp.Matrix([[1,0],[-c*dt/2*k, 1 - c*Fr*dt/2*k + c*c*dt*dt/4*k*k]])
    bracket_coeff = smp.Matrix([[1, -c*Fr*dt/2*k],[0,1]]) * smp.Matrix([[1,0],[0, 1 + c*Fr*dt/2*k - c*c*dt*dt/4*k*k]]).inv()
    ampl_mat = bracket_coeff * (term_1 + term_2) + smp.Matrix([[0,-c/2*dt*k],[0,0]])
    return ampl_mat

def op_split_2(ampl_mat_prev_iter):
    coeff_1 = smp.Matrix([[0,0],[-c/2*dt*k,0]]) * smp.Matrix([[1+c*Fr/2*dt*k,0],[0,1]]).inv() * smp.Matrix([[1-c*Fr/2*dt*k,0],[0,1]])
    coeff_2 = smp.Matrix([[1,0],[-c*dt/2*k, 1 - c*Fr*dt/2*k + c*c*dt*dt/4*k*k]])
    term_1 = coeff_1 + coeff_2
    term_2 = smp.Matrix([[0,0],[0,-c*Fr*dt/2*k]]) * ampl_mat_prev_iter
    bracket_coeff = smp.Matrix([[1, -c*Fr*dt/2*k],[0,1]]) * smp.Matrix([[1,0],[0, 1 - c*c*dt*dt/4*k*k]]).inv()
    ampl_mat = bracket_coeff * (term_1 + term_2) + smp.Matrix([[0,-c/2*dt*k],[0,0]])
    return ampl_mat

def max_ev(lambda_1, lambda_2, c_a_val, c_g_val, nx):
    thetas = np.linspace(0,2*np.pi,nx)
    A_max = 0
    for theta_val in thetas:
        values = {theta: theta_val, c_a: c_a_val, c_g: c_g_val, dx: 2*np.pi/nx}

        l1 = complex(lambda_1.subs(values).evalf())
        l2 = complex(lambda_2.subs(values).evalf())

        A_theta = max(abs(l1), abs(l2))
        A_max = max(A_max, A_theta)

    return A_max

def find_A_max(lambda_1, lambda_2, n_c_vals, nx):
    c_a_vals = np.linspace(0.01,2.01,n_c_vals)
    c_g_vals = np.linspace(0.01,2.01,n_c_vals)

    A = np.zeros((len(c_a_vals), len(c_g_vals)))

    for i, c_a_val in enumerate(c_a_vals):
        for j, c_g_val in enumerate(c_g_vals):
            A[i,j] = max_ev(lambda_1,lambda_2, c_a_val, c_g_val, nx)
            
    print(A)
    return c_a_vals, c_g_vals, A

def plot_3d(lambda_1,lambda_2,plot_name):
    
    c_a_vals, c_g_vals, A = find_A_max(lambda_1, lambda_2, 10, 100)
            
    CA, CG = np.meshgrid(c_a_vals, c_g_vals, indexing='ij')
    fig = plt.figure(figsize=(15,15))
    ax = plt.axes(projection='3d')
    ax.set_xlabel('c_a')
    ax.set_ylabel('c_g')
    ax.set_zlabel('amplification factor')
    colours = np.empty(A.shape, dtype=object)
    colours[A <= 1] = 'green' #stable
    colours[A > 1]  = 'red' #unstable
    ax.plot_surface(CA, CG, A, acecolors=colours, shade=False)
    ax.contour(CA, CG, A, levels=[1.0], colors='black')
    plt.savefig("stability_plots/"+plot_name+"_3d.png")
    plt.cla()

def plot_2d(lambda_1,lambda_2,plot_name):
    
    c_a_vals, c_g_vals, A = find_A_max(lambda_1, lambda_2, 10, 100)
    
    CA, CG = np.meshgrid(c_a_vals, c_g_vals, indexing='ij')
    fig = plt.figure(figsize=(15,15))
    ax = plt.axes()
    ax.set_xlabel('c_a')
    ax.set_ylabel('c_g')
    ax.contourf(CA, CG, A, levels=[0.5, 1.0000000001, A.max()],colors=['green', 'red'])
    ax.contour(CA, CG, A, levels=[1.0], colors='black')
    #fig.colorbar(cf, label='amplification factor')
    plt.tight_layout()
    plt.savefig("stability_plots/"+plot_name+"_2d.png")
    plt.cla()

def plot_for_c_a(lambda_1, lambda_2, c_a_val, nx, plot_name):
    c_g_vals = np.linspace(0,2,20)
    A = np.zeros(len(c_g_vals))
    for j, c_g_val in enumerate(c_g_vals):
        A[j] = max_ev(lambda_1,lambda_2, c_a_val, c_g_val, nx)
    
    plt.plot(c_g_vals,A)
    plt.savefig("stability_plots/"+plot_name+"_const_c_a.png")
    plt.cla()

def plot_for_c_g(lambda_1, lambda_2, c_g_val, nx, plot_name):
    c_a_vals = np.linspace(0,2,20)
    A = np.zeros(len(c_a_vals))
    for i, c_a_val in enumerate(c_a_vals):
        A[i] = max_ev(lambda_1,lambda_2, c_a_val, c_g_val, nx)
    
    plt.plot(c_a_vals,A)
    plt.savefig("stability_plots/"+plot_name+"_const_c_g.png")
    plt.cla()


def run_segregated():
    for its in range(1,4):
        A_mat = id_mat
        for i in range(its):
            A_mat = segregated(A_mat)

        A_mat = A_mat.subs(substitutions)
        print(A_mat)
        det = A_mat.det()
        tr = A_mat.trace()
        sqrt = smp.sqrt(tr*tr - 4*det)
        lambda_1 = 1/2 * (tr + sqrt)
        lambda_2 = 1/2 * (tr - sqrt)

        plot_2d(lambda_1,lambda_2,"segregated_"+str(its))

def run_semi_implicit():
    for its in range(1,4):
        A_mat = id_mat
        for i in range(its):
            A_mat = semi_implicit(A_mat)

        A_mat = A_mat.subs(substitutions)
        print(A_mat)
        det = A_mat.det()
        tr = A_mat.trace()
        sqrt = smp.sqrt(tr*tr - 4*det)
        lambda_1 = 1/2 * (tr + sqrt)
        lambda_2 = 1/2 * (tr - sqrt)

        plot_2d(lambda_1,lambda_2,"semi_implicit_"+str(its))


for its in range(1,4):
    A_mat = id_mat
    for i in range(its):
        A_mat = op_split_1(A_mat)

    A_mat = A_mat.subs(substitutions)
    print(A_mat)
    det = A_mat.det()
    tr = A_mat.trace()
    sqrt = smp.sqrt(tr*tr - 4*det)
    lambda_1 = 1/2 * (tr + sqrt)
    lambda_2 = 1/2 * (tr - sqrt)
    
    plot_for_c_g(lambda_1, lambda_2, 0, 100, "op_split_1")
    
    #plot_2d(lambda_1,lambda_2,"op_split_1_"+str(its))


for its in range(1,4):
    A_mat = id_mat
    for i in range(its):
        A_mat = op_split_2(A_mat)

    A_mat = A_mat.subs(substitutions)
    print(A_mat)
    det = A_mat.det()
    tr = A_mat.trace()
    sqrt = smp.sqrt(tr*tr - 4*det)
    lambda_1 = 1/2 * (tr + sqrt)
    lambda_2 = 1/2 * (tr - sqrt)
    
    print(max_ev(lambda_1,lambda_2, 0, 0, 100))

    #plot_2d(lambda_1,lambda_2,"op_split_2_"+str(its))
    
    
#print(max_ev(lambda_1,lambda_2, 0.01, 0.1, 100))

#plot_for_c_a(lambda_1, lambda_2, 0.5, 100)

vals = {
    theta: np.pi/2,
    c_a: 1,
    c_g: 1,
    dx: 2*np.pi/100
}

A_num = np.array(A_mat.subs(vals), dtype=np.complex128)
det = np.linalg.det(A_num)
tr = np.trace(A_num)
sqrt = np.sqrt(tr*tr - 4*det)
lambda_1 = 1/2 * (tr + sqrt)
lambda_2 = 1/2 * (tr - sqrt)
#print(A_num)
#print(np.linalg.eigvals(A_num))
#print(lambda_1,lambda_2)
