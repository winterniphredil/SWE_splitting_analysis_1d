import sympy as smp
from sympy import sin, symbols, Abs, Max, cos, E, I, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import ListedColormap, TwoSlopeNorm, BoundaryNorm
from matplotlib.lines import Line2D

matplotlib.rcParams.update({'font.size': 18})

y = symbols('y')
a,g,f = symbols('a g f', real = True)
dt = symbols('dt', real = True)
n_c_vals = 20

scatter = False
contour = False
stability = False
accuracy = False

if "-scatter" in sys.argv[3:]:scatter = True
if "-contour" in sys.argv[3:]:contour = True
if "-stability" in sys.argv[3:]:stability = True
if "-accuracy" in sys.argv[3:]:accuracy = True

def generate_A_mat_f_g(A_exp, c_f_vals, c_g_vals):
    """
    Calculates the amplification factors for a range of f,g values with a=0

    Args:
        A_exp (expr): the amplification factor already substituted with a=0
        c_f_vals (np vector): the values to substitute for f
        c_g_vals (np vector): the values to substitute for g
    """
    A = np.zeros((len(c_f_vals), len(c_g_vals)))
    
    for i in range(n_c_vals):
        for j in range(n_c_vals):
            A_sub = A_exp.subs({f: c_f_vals[i], g: c_g_vals[j]}).expand()
            A[i,j] = smp.Abs(A_sub)
    
    return A

def plot_f_g_0d(A_exp):
    """
    Plots the stability region for a = 0

    Args:
        A_exp (expr): the amplification factor already substituted with a=0
    """
    
    fig = plt.figure(figsize=(15,15))
    ax = plt.axes()
    ax.set_xlabel(r'$c_f$')
    ax.set_ylabel(r'$c_g$')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    cLevels=np.arange(0, 2.1, 0.1)
    cNorm = TwoSlopeNorm(1.,vmin=0., vmax=2.)
    
    c_f_vals = np.linspace(0.0,3.0,n_c_vals)
    c_g_vals = np.linspace(0.0,3.0,n_c_vals)
    
    A = generate_A_mat_f_g(A_exp, c_f_vals, c_g_vals)
    
    cf, cg = np.meshgrid(c_f_vals, c_g_vals, indexing='ij')
    c_f = ax.contourf(cf, cg, A, cLevels, cmap='bwr', extend='max', norm=cNorm)
    plt.colorbar(c_f, orientation='horizontal')
    plt.contour(cf, cg, A, levels=[1.000000001], colors='black', linewidths=1)
    plt.axvline(1.0, color='black', linestyle=':', linewidth=1)
    plt.axhline(1.0, color='black', linestyle=':', linewidth=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
   
   
def generate_A_mat_a_f(A_exp, c_a_vals, c_f_vals):
    """
    Calculates the amplification factors for a range of a,f values with g=0

    Args:
        A_exp (expr): the amplification factor already substituted with g=0
        c_a_vals (np vector): the values to substitute for a
        c_f_vals (np vector): the values to substitute for f
    """
    A = np.zeros((len(c_a_vals), len(c_f_vals)))
    
    for i in range(n_c_vals):
        for j in range(n_c_vals):
            A_sub = A_exp.subs({f: c_f_vals[j], a: c_a_vals[i]}).expand()
            #print(A_sub)
            A[i,j] = smp.Abs(A_sub)
    
    return A

def plot_a_f_0d(A_exp):
    """
    Plots the stability region for g = 0

    Args:
        A_exp (expr): the amplification factor already substituted with g=0
    """
    
    fig = plt.figure(figsize=(15,15))
    ax = plt.axes()
    ax.set_xlabel(r'$c_a$')
    ax.set_ylabel(r'$c_f$')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    cLevels=np.arange(0, 2.1, 0.1)
    cNorm = TwoSlopeNorm(1.,vmin=0., vmax=2.)
    
    c_f_vals = np.linspace(0.0,3.0,n_c_vals)
    c_a_vals = np.linspace(0.0,3.0,n_c_vals)
    
    A = generate_A_mat_a_f(A_exp, c_a_vals, c_f_vals)
    
    ca, cf = np.meshgrid(c_a_vals, c_f_vals, indexing='ij')
    c_f = ax.contourf(ca, cf, A, cLevels, cmap='bwr', extend='max', norm=cNorm)
    plt.colorbar(c_f, orientation='horizontal')
    plt.contour(ca, cf, A, levels=[1.000000001], colors='black', linewidths=1)
    plt.axvline(1.0, color='black', linestyle=':', linewidth=1)
    plt.axhline(1.0, color='black', linestyle=':', linewidth=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def generate_A_mat_a_g(A_exp, c_a_vals, c_g_vals):
    """
    Calculates the amplification factors for a range of a,g values with f=0

    Args:
        A_exp (expr): the amplification factor already substituted with f=0
        c_a_vals (np vector): the values to substitute for a
        c_g_vals (np vector): the values to substitute for g
    """
    A = np.zeros((len(c_a_vals), len(c_g_vals)))
    
    for i in range(n_c_vals):
        for j in range(n_c_vals):
            A_sub = A_exp.subs({a: c_a_vals[i], g: c_g_vals[j]}).expand()
            #print(smp.Abs(A_sub))
            A[i,j] = smp.Abs(A_sub)
    
    return A

def plot_a_g_0d(A_exp):
    """
    Plots the stability region for f = 0

    Args:
        A_exp (expr): the amplification factor already substituted with f=0
    """
    
    fig = plt.figure(figsize=(15,15))
    ax = plt.axes()
    ax.set_xlabel(r'$c_a$')
    ax.set_ylabel(r'$c_g$')
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    cLevels=np.arange(0, 2.1, 0.1)
    cNorm = TwoSlopeNorm(1.,vmin=0., vmax=2.)
    
    c_a_vals = np.linspace(0.0,3.0,n_c_vals)
    c_g_vals = np.linspace(0.0,3.0,n_c_vals)
    
    A = generate_A_mat_a_g(A_exp, c_a_vals, c_g_vals)
    
    ca, cg = np.meshgrid(c_a_vals, c_g_vals, indexing='ij')
    cf = ax.contourf(ca, cg, A, cLevels, cmap='bwr', extend='max', norm=cNorm)
    plt.colorbar(cf, orientation='horizontal')
    plt.contour(ca, cg, A, levels=[1.000000001], colors='black', linewidths=1)
    plt.axvline(1.0, color='black', linestyle=':', linewidth=1)
    plt.axhline(1.0, color='black', linestyle=':', linewidth=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def ampl_factor(a_tabl,g_tabl,f_tabl):
    """
    Calculates the amplification factor in terms of a, f, and g

    Args:
        a_tabl (2d array): the advection Butcher tableau
        g_tabl (2d array): the gravity Butcher tableau
        f_tabl (2d array): the Coriolis Butcher tableau
    """
    y = [1]
    
    for i in range(len(a_tabl)):
        this_y = 1
        
        for j in range(i+1):
            this_y += (I*a*a_tabl[i][j] + I*g*g_tabl[i][j] + I*f*f_tabl[i][j])*y[j]
            
        this_y /= 1 - (I*a*a_tabl[i][i+1] + I*g*g_tabl[i][i+1])
        y.append(this_y)
        
    return y[-1]


def accuracy(a_tabl,g_tabl,f_tabl):
    """
    Calculates and prints the difference between the analytic and numerical solution in ascending powers of dt

    Args:
        a_tabl (2d array): the advection Butcher tableau
        g_tabl (2d array): the gravity Butcher tableau
        f_tabl (2d array): the Coriolis Butcher tableau
    """
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
    Calculates and prints the advective and gravitational Courant numbers

    Args:
        a_tabl (2d array): the advection Butcher tableau
        g_tabl (2d array): the gravity Butcher tableau
        f_tabl (2d array): the Coriolis Butcher tableau
    """
    if accuracy: accuracy(a_tabl,g_tabl,f_tabl)
    if stability:
        ampl = ampl_factor(a_tabl,g_tabl,f_tabl)
        return ampl

def plot_3d(ampl):
    """
    Plots a 3d scatter grid of stable / unstable points

    Args:
        ampl (expr): the expression for the amplification factor in terms of a, f, and g
    """
    av = np.linspace(0, 3, 60)
    gv = np.linspace(0, 3, 60)
    fv = np.linspace(0, 3, 300)

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

    if contour:
        A, G = np.meshgrid(av, gv, indexing='ij')
        
        for i, f_value in enumerate(fv):
            absA = np.abs(ampl_func(A, G, f_value))
            ax.contour(A, G, absA, levels=[1.0000001], zdir='z', offset=f_value, color='black', linewidths=2)

    ax.set_xlabel(r'$a$')
    ax.set_ylabel(r'$g$')
    ax.set_zlabel(r'$f$')
    
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_zlim(0, 3)
    
    ax.set_title(r'Stability regions: $|A|\leq1$ stable, $|A|>1$ unstable')

    ax.legend()

    plt.tight_layout()
    plt.show()


# CORIOLIS INSIDE

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

print("Coriolis inside:\n")
ampl = interpret_tableau(a_bt,g_bt,f_bt)
if stability:
    print(ampl)
    plot_a_f_0d(ampl.subs({g:0}))
    plot_a_g_0d(ampl.subs({f:0}))
    plot_f_g_0d(ampl.subs({a:0}))

if scatter or contour:plot_3d(ampl)

# CORIOLIS OUTSIDE - OLD

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
    [0,   0,   0,   0,   1/2, 1/2, 0],  # n+1
]

# CORIOLIS OUTSIDE - NEW

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


print("Coriolis outside:\n")
ampl = interpret_tableau(a_bt,g_bt,f_bt)
if stability:
    print(ampl)
    plot_a_f_0d(ampl.subs({g:0}))
    plot_a_g_0d(ampl.subs({f:0}))
    plot_f_g_0d(ampl.subs({a:0}))

if scatter or contour:plot_3d(ampl)
