import sympy as smp
from sympy import symbols, I
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import TwoSlopeNorm
import sys

matplotlib.rcParams.update({'font.size': 18})

# Symbolic Courant numbers used throughout the stability analysis
# a = advective Courant number
# g = gravity-wave Courant number
# f = Coriolis Courant number
a,g,f = symbols('a g f', real = True)
n_c_vals = 20
C_MIN = 0.0
C_MAX = 3.0


args = sys.argv[1:]

scatter = "-scatter" in args
contour = "-contour" in args
stability = "-stability" in args
accuracy = "-accuracy" in args
inside = "-inside" in args
outside = "-outside" in args
new_cn = "-new-cn" in args
new_ssp = "-new-ssp" in args

plots_title = sys.argv[-1]


def generate_A_mat(A_exp, x_symbol, x_vals, y_symbol, y_vals):
    A = np.zeros((len(x_vals), len(y_vals)))

    for i, x_val in enumerate(x_vals):
        for j, y_val in enumerate(y_vals):
            A_sub = A_exp.subs({
                x_symbol: x_val,
                y_symbol: y_val
            }).expand()
            A[i, j] = float(smp.Abs(A_sub))

    return A
    


def plot_f_g_0d(A_exp):
    """
    Plots the stability region for a = 0

    Args:
        A_exp (expr): the amplification factor already substituted with a=0
    """
    
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_xlabel(r'$c_f$')
    ax.set_ylabel(r'$c_g$')
    ax.set_xlim(C_MIN, C_MAX)
    ax.set_ylim(C_MIN, C_MAX)
    cLevels=np.arange(0, 2.1, 0.1)
    cNorm = TwoSlopeNorm(1.,vmin=0., vmax=2.)
    
    c_f_vals = np.linspace(C_MIN, C_MAX, n_c_vals)
    c_g_vals = np.linspace(C_MIN, C_MAX, n_c_vals)
    
    A = generate_A_mat(A_exp, f, c_f_vals, g, c_g_vals)
    
    cf, cg = np.meshgrid(c_f_vals, c_g_vals, indexing='ij')
    c_f = ax.contourf(cf, cg, A, cLevels, cmap='bwr', extend='max', norm=cNorm)
    plt.colorbar(c_f, orientation='horizontal', fraction=0.046, pad=0.1)
    plt.contour(cf, cg, A, levels=[1.000000001], colors='black', linewidths=1)
    plt.axvline(1.0, color='black', linestyle=':', linewidth=1)
    plt.axhline(1.0, color='black', linestyle=':', linewidth=1)
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(plots_title+"_no_a.png")
    plt.close()
   


def plot_a_f_0d(A_exp):
    """
    Plots the stability region for g = 0

    Args:
        A_exp (expr): the amplification factor already substituted with g=0
    """
    
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_xlabel(r'$c_a$')
    ax.set_ylabel(r'$c_f$')
    ax.set_xlim(C_MIN, C_MAX)
    ax.set_ylim(C_MIN, C_MAX)
    cLevels=np.arange(0, 2.1, 0.1)
    cNorm = TwoSlopeNorm(1.,vmin=0., vmax=2.)
    
    c_f_vals = np.linspace(C_MIN, C_MAX, n_c_vals)
    c_a_vals = np.linspace(C_MIN, C_MAX, n_c_vals)
    
    A = generate_A_mat(A_exp, a, c_a_vals, f, c_f_vals)
    
    ca, cf = np.meshgrid(c_a_vals, c_f_vals, indexing='ij')
    c_f = ax.contourf(ca, cf, A, cLevels, cmap='bwr', extend='max', norm=cNorm)
    plt.colorbar(c_f, orientation='horizontal', fraction=0.046, pad=0.1)
    plt.contour(ca, cf, A, levels=[1.000000001], colors='black', linewidths=1)
    plt.axvline(1.0, color='black', linestyle=':', linewidth=1)
    plt.axhline(1.0, color='black', linestyle=':', linewidth=1)
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(plots_title+"_no_g.png")
    plt.close()


def plot_a_g_0d(A_exp):
    """
    Plots the stability region for f = 0

    Args:
        A_exp (expr): the amplification factor already substituted with f=0
    """
    
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.set_xlabel(r'$c_a$')
    ax.set_ylabel(r'$c_g$')
    ax.set_xlim(C_MIN, C_MAX)
    ax.set_ylim(C_MIN, C_MAX)
    cLevels=np.arange(0, 2.1, 0.1)
    cNorm = TwoSlopeNorm(1.,vmin=0., vmax=2.)
    
    c_a_vals = np.linspace(C_MIN, C_MAX, n_c_vals)
    c_g_vals = np.linspace(C_MIN, C_MAX, n_c_vals)
    
    A = generate_A_mat(A_exp, a, c_a_vals, g, c_g_vals)
    
    ca, cg = np.meshgrid(c_a_vals, c_g_vals, indexing='ij')
    cf = ax.contourf(ca, cg, A, cLevels, cmap='bwr', extend='max', norm=cNorm)
    plt.colorbar(cf, orientation='horizontal', fraction=0.046, pad=0.1)
    plt.contour(ca, cg, A, levels=[1.000000001], colors='black', linewidths=1)
    plt.axvline(1.0, color='black', linestyle=':', linewidth=1)
    plt.axhline(1.0, color='black', linestyle=':', linewidth=1)
    ax.set_aspect('equal', adjustable='box')
    plt.savefig(plots_title+"_no_f.png")
    plt.close()

def ampl_factor(a_tabl,g_tabl,f_tabl):
    """
    Calculates the amplification factor in terms of a, f, and g

    Args:
        a_tabl (2d array): the advection Butcher tableau
        g_tabl (2d array): the gravity Butcher tableau
        f_tabl (2d array): the Coriolis Butcher tableau
    """
    
    # amplification factor for current stage
    y = [1]
    
    for i in range(len(a_tabl)):
        this_y = 1
        
        # explicit part
        for j in range(i+1):
            this_y += (I*a*a_tabl[i][j] + I*g*g_tabl[i][j] + I*f*f_tabl[i][j])*y[j]
            
        # implicit part
        this_y /= 1 - (I*a*a_tabl[i][i+1] + I*g*g_tabl[i][i+1])
        y.append(this_y)
        
    return y[-1]



def ampl_factor_nl(a_tabl,g_tabl,f_tabl):
    """
    Calculates the amplification factor of energy in terms of a, f, and g

    Args:
        a_tabl (2d array): the advection Butcher tableau
        g_tabl (2d array): the gravity Butcher tableau
        f_tabl (2d array): the Coriolis Butcher tableau
    """
    
    # amplification factor for current stage
    x = [1]
    h = [1]
    
    for i in range(len(a_tabl)):
        this_x = 1
        this_h = 1
        
        # explicit part
        for j in range(i+1):
            this_x += (I*a*a_tabl[i][j]*x[j]*x[j] + I*g*g_tabl[i][j]*h[j] + I*f*f_tabl[i][j]*x[j])
            this_h += (I*a*a_tabl[i][j]*h[j]*x[j] + I*g*g_tabl[i][j]*x[j]*h[j])
            
        # implicit part
        this_x /= 1 - (I*a*a_tabl[i][i+1]*x[i-1] + I*g*g_tabl[i][i+1])
        this_h /= 1 - (I*a*a_tabl[i][i+1]*x[i-1] + I*g*g_tabl[i][i+1]*h[i-1])
        
        x.append(this_x)
        h.append(this_h)
    
    energy = 1/2 * x[-1]**2 * h[-1] + 1/2 * h[-1]**2
    
    return energy


def print_accuracy(a_tabl,g_tabl,f_tabl):
    """
    Calculates and prints the difference between the analytic and numerical solution in ascending powers of dt

    Args:
        a_tabl (2d array): the advection Butcher tableau
        g_tabl (2d array): the gravity Butcher tableau
        f_tabl (2d array): the Coriolis Butcher tableau
    """
    dt = symbols('dt', real = True)
    u_0, v_0, h_0 = symbols('u_0 v_0 h_0', real=True)
    x_0 = smp.Matrix([u_0,v_0,h_0])
    x = [x_0]
    M_a = -I*a * smp.eye(3)

    M_g = -I*g * smp.Matrix([
        [0, 0, 1],
        [0, 0, 1],
        [1, 1, 0]
    ])

    M_f = f * smp.Matrix([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ])
    
    for i in range(len(a_tabl)):
        this_x = x_0
        for j in range(i+1):
            this_x += dt*(a_tabl[i][j]*M_a + g_tabl[i][j]*M_g + f_tabl[i][j]*M_f)@x[j]
        this_x = (smp.eye(3) - dt*(a_tabl[i][i+1]*M_a + g_tabl[i][i+1]*M_g)).inv()@this_x
        x.append(this_x)
    
    A_matrix = smp.simplify(x[-1].jacobian([u_0, v_0, h_0]))
    M = M_a + M_g + M_f
    
    diff = A_matrix - smp.eye(3) - M*dt - M**2 * dt**2/2 - M**3 * dt**3/6
    for i in range(3):
        for j in range(3):
            print(i," ",j," : ",smp.series(diff[i,j],dt,0,4),"\n") 


def interpret_tableau(a_tabl,g_tabl,f_tabl):
    """
    Performs the requested accuracy and stability analysis.

    Args:
        a_tabl (2d array): the advection Butcher tableau
        g_tabl (2d array): the gravity Butcher tableau
        f_tabl (2d array): the Coriolis Butcher tableau
    """
    if accuracy: print_accuracy(a_tabl,g_tabl,f_tabl)
    if stability:
        ampl = ampl_factor_nl(a_tabl,g_tabl,f_tabl)
        return ampl

def plot_3d(ampl):
    """
    Plots a 3d scatter grid of stable / unstable points

    Args:
        ampl (expr): the expression for the amplification factor in terms of a, f, and g
    """
    av = np.linspace(C_MIN, C_MAX, n_c_vals)
    gv = np.linspace(C_MIN, C_MAX, n_c_vals)
    fv = np.linspace(C_MIN, C_MAX, n_c_vals*10)

    ampl_func = smp.lambdify((a, g, f), ampl, 'numpy')
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if scatter:
        A, G, F = np.meshgrid(av, gv, fv, indexing='ij')
        absA = np.abs(ampl_func(A, G, F))
        stable = absA <= 1
        unstable = absA > 1
        
        
        ax.scatter(A[stable], G[stable], F[stable], s=1, color='blue', label=r'Stable: $|A|\leq1$')
        ax.scatter(A[unstable], G[unstable], F[unstable], s=1, color='red', label=r'Unstable: $|A|>1$')
    
    
    # Plot |A|=1 as a sequence of constant-f contour slices.
    if contour:
        A, G = np.meshgrid(av, gv, indexing='ij')
        
        for f_value in fv:
            absA = np.abs(ampl_func(A, G, f_value))
            ax.contour(A, G, absA, levels=[1.0000001], zdir='z', offset=f_value, colors='black', linewidths=2)

    ax.set_xlabel(r'$a$')
    ax.set_ylabel(r'$g$')
    ax.set_zlabel(r'$f$')
    
    ax.set_xlim(C_MIN, C_MAX)
    ax.set_ylim(C_MIN, C_MAX)
    ax.set_zlim(C_MIN, C_MAX)

    #ax.legend()

    plt.tight_layout()
    plt.savefig(plots_title+"_3d_plot.png")
    plt.close()

def inside_tbl():
    """
    Tableaus with original Coriolis treatment, within the other operators
    """
    a_bt = [
        [1/2, 0,   0,   0,   0,   0,   0],  # ag1
        [1/2, 0,   0,   0,   0,   0,   0],  # f1
        [1/2, 0,   0,   0,   0,   0,   0],  # f2
        [1/2, 0,   0,   0,   0,   0,   0],  # f3
        [1/2, 0,   0,   0,   0,   1/2, 0],  # a2
        [1/2, 0,   0,   0,   0,   1/2, 0],  # n+1
    ]

    g_bt = [
        [1/2, 0,   0,   0,   0,   0,   0],  # ag1
        [1/2, 0,   0,   0,   0,   0,   0],  # f1
        [1/2, 0,   0,   0,   0,   0,   0],  # f2
        [1/2, 0,   0,   0,   0,   0,   0],  # f3
        [1/2, 0,   0,   0,   0,   0,   0],  # a2
        [1/2, 0,   0,   0,   0,   0, 1/2],  # n+1
    ]

    f_bt = [
        [0,   0,   0,   0,   0,   0,   0],  # ag1
        [0,   1/2, 0,   0,   0,   0,   0],  # f1
        [0,   0,   1/2, 0,   0,   0,   0],  # f2
        [0,   0,   0,   1,   0,   0,   0],  # f3
        [0,   0,   0,   1,   0,   0,   0],  # a2
        [0,   0,   0,   1,   0,   0,   0],  # n+1
    ]
    
    return a_bt, g_bt, f_bt


def outside_tbl():
    """
    Tableaus with original Coriolis treatment outside the other operators
    """
    a_bt = [
        [0,   0,   0,   0,   0,   0,   0],  # 1c
        [0,   0,   0,   0,   0,   0,   0],  # 2c
        [0,   0,   1/2, 0,   0,   0,   0],  # 1a
        [0,   0,   1/2, 0,   0,   0,   0],  # g
        [0,   0,   1/2, 0,   0,   1/2, 0],  # 2a
        [0,   0,   1/2, 0,   0,   1/2, 0],  # n+1
    ]

    g_bt = [
        [0,   0,   0,   0,   0,   0,   0],  # 1c
        [0,   0,   0,   0,   0,   0,   0],  # 2c
        [0,   0,   0,   0,   0,   0,   0],  # 1a
        [0,   0,   0,   1/2, 1/2, 0,   0],  # g
        [0,   0,   0,   1/2, 1/2, 0,   0],  # 2a
        [0,   0,   0,   1/2, 1/2, 0,   0],  # n+1
    ]

    f_bt = [
        [1/2, 0,   0,   0,   0,   0,   0],  # 1c
        [0,   1/2, 0,   0,   0,   0,   0],  # 2c
        [0,   1/2, 0,   0,   0,   0,   0],  # 1a
        [0,   1/2, 0,   0,   0,   0,   0],  # g
        [0,   1/2, 0,   0,   0,   0,   0],  # 2a
        [0,   0,   0,   1/2, 0,   1/2, 0],  # n+1
    ]
    
    return a_bt, g_bt, f_bt


def new_CN_tbl():
    """
    Tableaus with improved Coriolis treatment outside the other operators
    """
    a_bt = [
        [0,   0,   0,   0,   0,   0,   0,   0,   0],  # 1c
        [0,   0,   0,   0,   0,   0,   0,   0,   0],  # 2c
        [0,   0,   0,   0,   0,   0,   0,   0,   0],  # 3c
        [0,   0,   0,   1/2, 0,   0,   0,   0,   0],  # 1a
        [0,   0,   0,   1/2, 0,   0,   0,   0,   0],  # g
        [0,   0,   0,   1/2, 0,   0,   1/2, 0,   0],  # 2a
        [0,   0,   0,   1/2, 0,   0,   1/2, 0,   0],  # 4c
        [0,   0,   0,   1/2, 0,   0,   1/2, 0,   0],  # n+1
    ]

    g_bt = [
        [0,   0,   0,   0,   0,   0,   0,   0,   0],  # 1c
        [0,   0,   0,   0,   0,   0,   0,   0,   0],  # 2c
        [0,   0,   0,   0,   0,   0,   0,   0,   0],  # 3c
        [0,   0,   0,   0,   0,   0,   0,   0,   0],  # 1a
        [0,   0,   0,   0,   1/2, 1/2, 0,   0,   0],  # g
        [0,   0,   0,   0,   1/2, 1/2, 0,   0,   0],  # 2a
        [0,   0,   0,   0,   1/2, 1/2, 0,   0,   0],  # 4c
        [0,   0,   0,   0,   1/2, 1/2, 0,   0,   0],  # n+1
    ]

    f_bt = [
        [1/2, 0,   0,   0,   0,   0,   0,   0,   0],  # 1c
        [0,   1/2, 0,   0,   0,   0,   0,   0,   0],  # 2c
        [0,   0,   1/2, 0,   0,   0,   0,   0,   0],  # 3c
        [0,   0,   1/2, 0,   0,   0,   0,   0,   0],  # 1a
        [0,   0,   1/2, 0,   0,   0,   0,   0,   0],  # g
        [0,   0,   1/2, 0,   0,   0,   0,   0,   0],  # 2a
        [0,   0,   0,   0,   0,   1/2, 0,   0,   0],  # 4c
        [0,   0,   0,   1/2, 0,   0,   1/2, 0,   0],  # n+1
    ]
    return a_bt, g_bt, f_bt
    
def new_2_tbl():
    """
    Tableaus for Hilary's propsed stable CN scheme with original Coriolis treatment
    """
    a_bt = [
        [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],  # 1c
        [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],  # 2c
        [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],  # 3c
        [0,   0,   0,   1/2, 0,   0,   0,   0,   0,   0],  # 1a
        [0,   0,   0,   1/2, 0,   0,   0,   0,   0,   0],  # g
        [0,   0,   0,   1/2, 0,   0,   0,   0,   0,   0],  # 2a
        [0,   0,   0,   1/2, 0,   0,   0, 1/2,   0,   0],  # 4c
        [0,   0,   0,   1/2, 0,   0,   0, 1/2,   0,   0],  # n+1
        [0,   0,   0,   1/2, 0,   0,   0, 1/2,   0,   0],  # n+1
    ]

    g_bt = [
        [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],  # 1c
        [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],  # 2c
        [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],  # 3c
        [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],  # 1a
        [0,   0,   0,   0,   1/2, 0,   0,   0,   0,   0],  # g
        [0,   0,   0,   0,   1/2, 0, 1/2,   0,   0,   0],  # 2a
        [0,   0,   0,   0,   1/2, 0, 1/2,   0,   0,   0],  # 4c
        [0,   0,   0,   0,   1/2, 0, 1/2,   0,   0,   0],  # n+1
        [0,   0,   0,   0,   1/2, 0, 1/2,   0,   0,   0],  # n+1
    ]

    f_bt = [
        [1/2, 0,   0,   0,   0,   0,   0,   0,   0,   0],  # 1c
        [0,   1/2, 0,   0,   0,   0,   0,   0,   0,   0],  # 2c
        [0,   0,   1/2, 0,   0,   0,   0,   0,   0,   0],  # 3c
        [0,   0,   1/2, 0,   0,   0,   0,   0,   0,   0],  # 1a
        [0,   0,   1/2, 0,   0,   0,   0,   0,   0,   0],  # g
        [0,   0,   1/2, 0,   0,   0,   0,   0,   0,   0],  # 2a
        [0,   0,   1/2, 0,   0,   0,   0,   0,   0,   0],  # 2a
        [0,   0,   0,   0,   0,   0,   1/2, 0,   0,   0],  # 4c
        [0,   0,   0,   1/2, 0,   0,   0,   1/2, 0,   0],  # n+1
    ]
    return a_bt, g_bt, f_bt

def new_3_tbl():
    """
    SSP2_332-based tableaus with original Coriolis treatment
    """
    a_bt = [
        [0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,   1/2, 0,   0,   0,   0,   0],
        [0,   0,   1/2, 0,   0,   0,   0,   0],
        [0,   0,   1/2, 0,   0,   0,   0,   0],
        [0,   0,   1/2, 0,   0,   0,   1/2, 0],
        [0,   0,   1/2, 0,   0,   0,   1/2, 0],
    ]

    g_bt = [
        [0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   1/4, 0,   0,   0],
        [0,   0,   0,   0,   2/3, 1/3, 0,   0],
        [0,   0,   0,   0,   2/3, 1/3, 0,   0],
        [0,   0,   0,   0,   2/3, 1/3, 0,   0],
    ]

    f_bt = [
        [1/2, 0,   0,   0,   0,   0,   0,   0],
        [0,   1/2, 0,   0,   0,   0,   0,   0],
        [0,   1/2, 0,   0,   0,   0,   0,   0],
        [0,   1/2, 0,   0,   0,   0,   0,   0],
        [0,   1/2, 0,   0,   0,   0,   0,   0],
        [0,   1/2, 0,   0,   0,   0,   0,   0],
        [0,   0,   1/2, 0,   0,   0,   1/2, 0],
    ]
    return a_bt, g_bt, f_bt

def new_SSP2_tbl():
    """
    SSP2_332-based tableaus with improved Coriolis treatment
    """
    a_bt = [
        [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,   0,   1/2, 0,   0,   0,   0,   0,   0],
        [0,   0,   0,   1/2, 0,   0,   0,   0,   0,   0],
        [0,   0,   0,   1/2, 0,   0,   0,   0,   0,   0],
        [0,   0,   0,   1/2, 0,   0,   0,   1/2, 0,   0],
        [0,   0,   0,   1/2, 0,   0,   0,   1/2, 0,   0],
    ]

    g_bt = [
        [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   1/4, 0,   0,   0,   0],
        [0,   0,   0,   0,   0,   2/3, 1/3, 0,   0,   0],
        [0,   0,   0,   0,   0,   2/3, 1/3, 0,   0,   0],
        [0,   0,   0,   0,   0,   2/3, 1/3, 0,   0,   0],
    ]

    f_bt = [
        [1/2, 0,   0,   0,   0,   0,   0,   0,   0,   0],
        [0,   1/2, 0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,   1/2, 0,   0,   0,   0,   0,   0,   0],
        [0,   0,   1/2, 0,   0,   0,   0,   0,   0,   0],
        [0,   0,   1/2, 0,   0,   0,   0,   0,   0,   0],
        [0,   0,   1/2, 0,   0,   0,   0,   0,   0,   0],
        [0,   0,   1/2, 0,   0,   0,   0,   0,   0,   0],
        [0,   0,   0,   1/2, 0,   0,   0,   1/2, 0,   0],
    ]
    return a_bt, g_bt, f_bt

if sum([inside, outside, new_cn, new_ssp]) != 1:
    raise ValueError("Select one tableau: -inside, -outside, -new-cn or -new-ssp")


# a_bt = advection Butcher tableau
# g_bt = gravity Butcher tableau
# f_bt = Coriolis Butcher tableau

if inside:
    a_bt, g_bt, f_bt = inside_tbl()
    print("Coriolis inside:\n")
    
if outside:
    a_bt, g_bt, f_bt = outside_tbl()
    print("Coriolis outside:\n")

if new_cn:
    a_bt, g_bt, f_bt = new_CN_tbl()
    print("New CN:\n")

if new_ssp:
    a_bt, g_bt, f_bt = new_SSP2_tbl()
    print("New SSP2:\n")

ampl = interpret_tableau(a_bt,g_bt,f_bt)

if stability:
    print("Amplification factor, f-axis (a=g=0): ", ampl.subs({a:0,g:0}))
    print("Amplification factor, g-axis (a=f=0): ", ampl.subs({a:0,f:0}))
    print("Amplification factor, a-axis (f=g=0): ", ampl.subs({f:0,g:0}))
    print(ampl)
    plot_a_f_0d(ampl.subs({g:0}))
    plot_a_g_0d(ampl.subs({f:0}))
    plot_f_g_0d(ampl.subs({a:0}))

if scatter or contour:plot_3d(ampl)



