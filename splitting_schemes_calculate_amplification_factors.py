import sympy as smp
from sympy import sin, symbols, Abs, Max
import numpy as np
import matplotlib.pyplot as plt

matrix_file = "ampl_matrices.txt"
ev_file = "ampl_eigenv.txt"

alpha = 1

k, Fr, c, c_a, c_g = symbols('k Fr c c_a c_g', real = True)
dt = symbols('dt', real = True)
a = symbols('alpha', real = True)
substitutions = {Fr: c_a/c_g, k: 1j*c_g/(dt*c), a: alpha}

id_mat = smp.Matrix([[1,0],[0,1]])


def semi_implicit(ampl_mat_prev_iter):
    prev_iter_contr = smp.Matrix([[0,0],[0,-a*c*Fr*dt*k]]) * ampl_mat_prev_iter
    prev_time_contr = smp.Matrix([[0,0],[-(1-a)*c*dt*k,-(1-a)*c*Fr*dt*k]])
    bracket_contr = smp.Matrix([[1,0],[-a*c*dt*k,1]]) * ( smp.Matrix([[1-(1-a)*c*Fr*dt*k,-(1-a)*c*dt*k],[0,1]]) + smp.Matrix([[-a*c*Fr*dt*k,0],[0,0]]) * ampl_mat_prev_iter )
    ampl_mat = smp.Matrix([[1,-a*c*dt*k],[0,1]]) * smp.Matrix([[1,0],[0,1-a*a*c*c*dt*dt*k*k]]).inv() * (prev_iter_contr + prev_time_contr + bracket_contr)
    return ampl_mat
    
def segregated(ampl_mat_prev_iter):
    bracket_1 = smp.Matrix([[1-(1-a)*c*Fr*dt*k,-(1-a)*c*dt*k],[0,1]]) + smp.Matrix([[0,-a*c*dt*k],[0,0]]) * ampl_mat_prev_iter
    bracket_coeff_1 = smp.Matrix([[0,0],[a*a*c*c*Fr*dt*dt*k*k,0]]) * smp.Matrix([[1+a*c*Fr*dt*k,-0],[0,1]]).inv()
    term_1 = smp.Matrix([[1,0],[a*(1-a)*c*c*Fr*dt*dt*k*k - c*dt*k , 1 - (1-a)*c*Fr*dt*k + a*(1-a)*c*c*dt*dt*k*k]])
    bracket_coeff_2 = smp.Matrix([[1, -a*c*Fr*dt*k],[0,1]]) * smp.Matrix([[1,0],[0, 1 + a*c*Fr*dt*k - a*a*c*c*dt*dt*k*k]]).inv()
    ampl_mat = bracket_coeff_2 * (term_1 + bracket_coeff_1 * bracket_1)
    return ampl_mat

def op_split_1(ampl_mat_prev_iter):
    term_1 = smp.Matrix([[0,0],[-a*c*dt*k,0]]) * smp.Matrix([[1+a*c*Fr*dt*k,0],[0,1]]).inv() * smp.Matrix([[1-(1-a)*c*Fr*dt*k,0],[0,1]])
    term_2 = smp.Matrix([[1,0],[-(1-a)*c*dt*k, 1 - (1-a)*c*Fr*dt*k + (1-a)*(1-a)*c*c*dt*dt*k*k]])
    bracket_coeff = smp.Matrix([[1, -a*c*Fr*dt*k],[0,1]]) * smp.Matrix([[1,0],[0, 1 + a*c*Fr*dt*k - a*a*c*c*dt*dt*k*k]]).inv()
    ampl_mat = bracket_coeff * (term_1 + term_2) + smp.Matrix([[0,-(1-a)*c*dt*k],[0,0]])
    return ampl_mat

def op_split_2(ampl_mat_prev_iter):
    coeff_1 = smp.Matrix([[0,0],[-a*c*dt*k,0]]) * smp.Matrix([[1+a*c*Fr*dt*k,0],[0,1]]).inv() * smp.Matrix([[1-(1-a)*c*Fr*dt*k,0],[0,1]])
    coeff_2 = smp.Matrix([[1,0],[-(1-a)*c*dt*k, 1 - (1-a)*c*Fr*dt*k + (1-a)*(1-a)*c*c*dt*dt*k*k]])
    term_1 = coeff_1 + coeff_2
    term_2 = smp.Matrix([[0,0],[0,-a*c*Fr*dt*k]]) * ampl_mat_prev_iter
    bracket_coeff = smp.Matrix([[1, -a*c*Fr*dt*k],[0,1]]) * smp.Matrix([[1,0],[0, 1 - a*a*c*c*dt*dt*k*k]]).inv()
    ampl_mat = bracket_coeff * (term_1 + term_2) + smp.Matrix([[0,-a*c*dt*k],[0,0]])
    return ampl_mat

def max_ev(lambda_1, lambda_2, c_a_val, c_g_val):
    values = {c_a: c_a_val, c_g: c_g_val}
    l1 = complex(lambda_1.subs(values).evalf())
    l2 = complex(lambda_2.subs(values).evalf())
    
    return max(abs(l1), abs(l2))

def plot_against_theta(lambda_1, lambda_2, c_a_val, c_g_val, nx, plot_name):
    thetas = np.linspace(0,2*np.pi,nx)
    A_theta = np.zeros(nx)
    
    for i in range(nx):
        theta_val = thetas[i]
        values = {theta: theta_val, c_a: c_a_val, c_g: c_g_val, dx: 2*np.pi/nx}

        l1 = complex(lambda_1.subs(values).evalf())
        l2 = complex(lambda_2.subs(values).evalf())

        A_theta[i] = max(abs(l1), abs(l2))
    
    plt.plot(thetas, A_theta)
    plt.xlabel("theta")
    plt.ylabel("A")
    plt.title("c_a = "+str(c_a_val)+", c_g = "+str(c_g_val))
    plt.savefig("stability_plots/"+plot_name+"_c_a_"+str(c_a_val)+"_c_g_"+str(c_g_val)+"_alpha_"+str(alpha)+".png")
        

def find_A_max(lambda_1, lambda_2, n_c_vals):
    c_a_vals = np.linspace(0.0,3.0,n_c_vals)
    c_g_vals = np.linspace(0.0,3.0,n_c_vals)

    A = np.zeros((len(c_a_vals), len(c_g_vals)))

    for i, c_a_val in enumerate(c_a_vals):
        for j, c_g_val in enumerate(c_g_vals):
            A[i,j] = max_ev(lambda_1,lambda_2, c_a_val, c_g_val)
            
    #print(A)
    return c_a_vals, c_g_vals, A

def plot_3d(lambda_1,lambda_2,plot_name):
    
    c_a_vals, c_g_vals, A = find_A_max(lambda_1, lambda_2, 10)
            
    CA, CG = np.meshgrid(c_a_vals, c_g_vals, indexing='ij')
    fig = plt.figure(figsize=(15,15))
    ax = plt.axes(projection='3d')
    ax.set_xlabel(r'$c_a$')
    ax.set_ylabel(r'$c_g$')
    ax.set_zlabel('amplification factor')
    colours = np.empty(A.shape, dtype=object)
    colours[A <= 1] = 'green' #stable
    colours[A > 1]  = 'red' #unstable
    ax.plot_surface(CA, CG, A, acecolors=colours, shade=False)
    ax.contour(CA, CG, A, levels=[1.0], colors='black')
    plt.savefig("stability_plots/"+plot_name+"_3d.png")
    plt.cla()

def plot_2d(lambda_1,lambda_2,plot_name):
    
    c_a_vals, c_g_vals, A = find_A_max(lambda_1, lambda_2, 25)
    
    CA, CG = np.meshgrid(c_a_vals, c_g_vals, indexing='ij')
    fig = plt.figure(figsize=(15,15))
    ax = plt.axes()
    ax.set_xlabel(r'$c_a$')
    ax.set_ylabel(r'$c_g$')
    ax.contourf(CA, CG, A, levels=[0., 1.0000000001, A.max()],colors=['green', 'red'])
    ax.contour(CA, CG, A, levels=[1.0], colors='black')
    #fig.colorbar(cf, label='amplification factor')
    plt.tight_layout()
    plt.savefig("stability_plots/"+plot_name+"_2d_alpha"+str(alpha)+".png")
    plt.cla()
    
def plot_continuous(lambda_1, lambda_2, plot_name):
    
    lambda_1_exp = smp.lambdify([c_a, c_g], Abs(lambda_1), 'numpy')
    lambda_2_exp = smp.lambdify([c_a, c_g], Abs(lambda_2), 'numpy')
    lambda_max = smp.lambdify([c_a, c_g], Max(Abs(lambda_1),Abs(lambda_2)), 'numpy')
    
    ca = np.linspace(0, 3, 150)
    cg = np.linspace(0, 3, 150)
    CA, CG = np.meshgrid(ca, cg)

    L1 = lambda_1_exp(CA, CG)
    L2 = lambda_2_exp(CA, CG)
    L_max = lambda_max(CA, CG)
    fig = plt.figure(figsize=(12, 5))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(CA, CG, L1, cmap='viridis')
    ax1.contour(CA, CG, L1, levels=[1.0], colors='black')
    ax1.set_title(r'$|\lambda_1|$')
    ax1.set_xlabel(r'$c_a$')
    ax1.set_ylabel(r'$c_g$')

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(CA, CG, L2, cmap='viridis')
    ax2.contour(CA, CG, L2, levels=[1.0], colors='black')
    ax2.set_title(r'$|\lambda_2|$')
    ax2.set_xlabel(r'$c_a$')
    ax2.set_ylabel(r'$c_g$')

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(CA, CG, L_max, cmap='viridis')
    ax3.contour(CA, CG, L_max, levels=[1.0], colors='black')
    ax3.set_title(r'$max(|\lambda|)$')
    ax3.set_xlabel(r'$c_a$')
    ax3.set_ylabel(r'$c_g$')

    plt.tight_layout()
    plt.savefig("stability_plots/"+plot_name+"_3d_cont.png")
    plt.cla()


def plot_for_c_a(lambda_1, lambda_2, c_a_val, plot_name):
    c_g_vals = np.linspace(0,2,20)
    A = np.zeros(len(c_g_vals))
    for j, c_g_val in enumerate(c_g_vals):
        A[j] = max_ev(lambda_1,lambda_2, c_a_val, c_g_val)
    
    plt.plot(c_g_vals,A)
    plt.savefig("stability_plots/"+plot_name+"_const_c_a.png")
    plt.cla()

def plot_for_c_g(lambda_1, lambda_2, c_g_val, plot_name):
    c_a_vals = np.linspace(0,2,20)
    A = np.zeros(len(c_a_vals))
    for i, c_a_val in enumerate(c_a_vals):
        A[i] = max_ev(lambda_1,lambda_2, c_a_val, c_g_val)
    
    plt.plot(c_a_vals,A)
    plt.savefig("stability_plots/"+plot_name+"_const_c_g.png")
    plt.cla()


def calc_evs(sch, its, A_mat):
    A_mat = A_mat.subs(substitutions)
    print(sch+" "+str(its)+"\n"+str(A_mat))
    with open(matrix_file,"a") as matfil:matfil.write(sch+" "+str(its)+"\n"+str(A_mat)+"\n\n")
    det = A_mat.det()
    tr = A_mat.trace()
    sqrt = smp.sqrt(tr*tr - 4*det)
    lambda_1 = 1/2 * (tr + sqrt)
    lambda_2 = 1/2 * (tr - sqrt)
    print(lambda_1)
    print(lambda_2)
    with open(ev_file,"a") as evfil:evfil.write(sch+" "+str(its)+"\n"+str(lambda_1)+"\n"+str(lambda_2)+"\n\n")
    return lambda_1, lambda_2

def run_segregated():
    A_mat = id_mat
    for its in range(1,4):
        A_mat = segregated(A_mat)
        
        lambda_1, lambda_2 = calc_evs("segregated", its, A_mat)

        plot_continuous(lambda_1,lambda_2,"segregated_"+str(its))

def run_semi_implicit():
    A_mat = id_mat
    for its in range(1,4):
        A_mat = semi_implicit(A_mat)

        lambda_1, lambda_2 = calc_evs("semi-implicit", its, A_mat)

        plot_continuous(lambda_1,lambda_2,"semi_implicit_"+str(its))

def run_op_split_1():
    A_mat = id_mat
    for its in range(1,4):
        A_mat = op_split_1(A_mat)

        lambda_1, lambda_2 = calc_evs("op-split 1", its, A_mat)
        
        #plot_for_c_g(lambda_1, lambda_2, 0, "op_split_1")
        
        plot_continuous(lambda_1,lambda_2,"op_split_1_"+str(its))


def run_op_split_2():
    A_mat = id_mat
    for its in range(1,4):
        A_mat = op_split_2(A_mat)

        lambda_1, lambda_2 = calc_evs("op-split 2", its, A_mat)
        
        #print(max_ev(lambda_1,lambda_2, 0, 0, 100))

        plot_continuous(lambda_1,lambda_2,"op_split_2_"+str(its))

def generate_theta_plot():
    A_mat = op_split_1(id_mat)
    A_mat = A_mat.subs(substitutions)
    print(A_mat)
    det = A_mat.det()
    tr = A_mat.trace()
    sqrt = smp.sqrt(tr*tr - 4*det)
    lambda_1 = 1/2 * (tr + sqrt)
    lambda_2 = 1/2 * (tr - sqrt) 
    plot_against_theta(lambda_1, lambda_2, 0.2, 0.2, 100, "op_split_1")


run_semi_implicit()
run_segregated()
run_op_split_1()
run_op_split_2()
