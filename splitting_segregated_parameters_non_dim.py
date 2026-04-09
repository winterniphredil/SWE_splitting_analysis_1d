from sympy import symbols, expand, cancel, re, E
from sympy import factor_terms
import sympy as smp
import numpy as np

file_to_write =  open("splitting_schemes_params_non_dim_exact.txt","w")

def taylor_exp_3(A):
    """
    Calculates the Taylor expansion up to third order of e^A for operator A (2x2 matrix)

    Args:
        A (2x2 matrix): operator 
    Returns:
        (expression): the Taylor expansion to third order
    """
    return smp.eye(2) + dt*A + dt**2/2 * A*A + dt**3/6 * A*A*A

def output(result,n,I):
    """
    Writes the result to the output file, can be modified to print output to console

    Args:
        result (string): what to write as output
        n (int): the number indicating the scheme used
        I (int): the number of iterations
    """
    output = "scheme = "+str(n)+"\niterations = "+str(I)+"\n"+result
    print(output)
    file_to_write.write(output)

def split_term(term): 
    """
    Takes a product of scalars and operators and separates it into the scalar coefficient and operator part

    Args:
        term (expression): the single term to be decomposed

    Returns:
        (expr): the scalar coefficient
        (expr): the operator part
    """
    coeff = 1
    this_term = 1

    for f in term.as_ordered_factors():
        if f in operators:
            this_term *= f
        else:
            coeff *= f

    return coeff, this_term

def collect_terms(expr): 
    """
    Splits an expression into operator terms and their coefficients

    Args:
        expr (expression): the expression to be decomposed

    Returns:
        (dictionary): keys are the operator terms and values are the corresponding coefficient
    """
    term_coeffs = {}

    for term in smp.expand(expr).as_ordered_terms():
        coeff, this_term = split_term(term)

        term_coeffs.setdefault(this_term, 0)
        term_coeffs[this_term] += coeff

    return term_coeffs

def simplify_commutators_for_3(expr):
    """
    Uses commutation relations to simplify an expression expr into a minimal number of terms using algebraic relations

    Args:
        expr (expression): the expression to be simplified

    Returns:
        (expression): the simplified expression
    """
    comm_rules = { 
        (A_h, A_u): lambda A,B: B*A,
        (A_h, G): lambda A,B: B*A - A_u*B + B*A_u,
        (A_h, F): lambda A,B: B*A - A_u*B + B*A_u
    } # commutation rules
    
    changed = True
    while changed: # iterate until all simplifications made
        changed = False
        new_terms = []
        for term in smp.expand(expr).as_ordered_terms(): # iterate through each additive term
            if term.is_Mul:
                factors = list(term.args)
            else:
                factors = [term]
            
            i = 0
            while i < len(factors)-1:
                pair = (factors[i], factors[i+1])
                if pair in comm_rules:
                    factors[i:i+2] = [comm_rules[pair](factors[i], factors[i+1])] # replace if in comm_rules
                    changed = True
                    factors = smp.expand(smp.Mul(*factors)).as_ordered_factors() if len(factors) > 1 else factors
                    i = -1  # restart iterations because may now contain new term from comm_rules
                i += 1
            new_terms.append(smp.Mul(*factors)) # add the simplified result of this term
        expr_new = sum(new_terms) # re-generate the expression with the simplified terms
        if expr_new != expr: # determine if this made a difference to the expression
            expr = expr_new
            changed = True
    return expr

A_h, A_u, F, G = symbols('A_h A_u F G', commutative=False)
dt = symbols('dt')




omega, k, g, U, H, u_0, h_0, Fr = symbols('omega k g U H u_0 h_0 Fr', real = True)
x = symbols('x', real = True)
t = symbols('t', real = True)


unsplit_op = [[(- A_u*u_0 - G*h_0) * dt], [(- A_h*h_0 - F*u_0) * dt]]  #the concurrent operator (negatives encoded elsewhere - CHECK??????)
unsplit_exp = [[u_0], [h_0]] + unsplit_op

exact_op = smp.Matrix([[-A_u,-G],[-F,-A_h]])
exact_exp = taylor_exp_3(exact_op) * smp.Matrix([u_0,h_0]) #the Taylor expansion of the concurrent operator application to second order

exact_exp = exact_exp.tolist()

operators = [A_h, A_u, F, G] #array of operators - this defines the general order in which they will be treated
#op_repl = {A_u: k*U, A_h: k*U, F: k*H, G:k*g} #dimensional
op_repl = {A_u: 1j*k, A_h: 1j*k, F: 1j*k, G:1j*k/(Fr**2)}

def rearrange_operator(operators_list):
    """
    Rearranges a semi-implicit operator into a single expression

    Args:
        operators_list (list): a list of tuples of the operators to be applied with their corresponding implicitness parameters

    Returns:
        (expression): the resulting expression
    """
    implicit = sum(O*theta for O,theta in operators_list)
    explicit = sum(O*(1-theta) for O,theta in operators_list)
    x = 1
    x = (explicit*x) * (1 - implicit)**(-1)
    return x

def calc_splitting_error(operator_sequence, step_ops, step_count, order, n, I):
    """
    Calculates the splitting error introduced by a given splitting scheme and outputs

    Args:
        operator_sequence (list): a list of lambda expressions to be evaluated sequentially
        step_ops (list): a list containing the initial values and 1 for following expressions which have yet to be evaluated
        step_count (int): the total number of steps, including the inital values
        order (int): the order in dt at which the error is to be evaluated
        n (int): respresenting the number of the scheme being used
        I (int): the number of iterations within each time update
    """
    
    for i in range(len(operator_sequence)):
        if step_ops[i] == 1:
            current_ops = operator_sequence[i]()
            step_ops[i] = rearrange_operator(current_ops)
    x_vec = [step_ops[step_count-1], step_ops[step_count-2]]
    diff = [smp.series(x_vec[i],dt,0,4).removeO() - exact_exp[i][0] for i in [0,1]]
    diff = [smp.collect(smp.expand(diff_i), dt).coeff(dt, order) for diff_i in diff] # only keep dt**2 term
    diff = [diff_i.subs(op_repl) for diff_i in diff]
    term_coeffs = [collect_terms(diff_i) for diff_i in diff]
    result = ""
    term_indicator = ["u","h"]
    for i in range(2):
        for word, coeff in term_coeffs[i].items():
           result += term_indicator[i] + " : " + str(re(smp.simplify(coeff)*E**(1j*k*x))) + "\n"
    diff = [diff_i.as_ordered_terms() for diff_i in diff]
    output(result, n, I)






for order_to_run in range(4):

    print("\n\n\nORDER = "+str(order_to_run)+"\n\n")
    file_to_write.write("\n\n\nORDER = "+str(order_to_run)+"\n\n")
    for I in range(1,5):
        no_steps = 3
        step_ops = []
        for i in range(no_steps*(I+1)): step_ops.append(1)

        #operator_sequence_1_1 = [
        #    lambda: [(1,0),(-dt*A_u,0),(-1/2*dt*G,0)],
        #    lambda: [(1,0),(-dt*A_h,0),(-1/2*dt*F,0),(-1/2*dt*F*step_ops[0],0),(1/4*dt*dt*F*G,1)],
        #    lambda: [(step_ops[0],0),(-1/2*dt*G*step_ops[1],0)]
        #]
        operator_sequence_1_1 = [lambda: [(u_0,0)], lambda: [(h_0,0)], lambda: [(u_0,0)]]
        for i in range(no_steps): step_ops[i] = operator_sequence_1_1[i]()[0][0]


        for i in range(0,I):
            operator_sequence_1_1.append(lambda i=i: [(step_ops[0],0),(-1/2*dt*A_u*(step_ops[0]+step_ops[(i+1)*no_steps-1]),0),(-1/2*dt*G*step_ops[1],0)])
            operator_sequence_1_1.append(lambda i=i: [(step_ops[1],0),(-1/2*dt*A_h*(step_ops[1]+step_ops[(i+1)*no_steps-2]),0),(-1/2*dt*F*step_ops[0],0),(-1/2*dt*F*step_ops[(i+1)*no_steps],0),(1/4*dt*dt*F*G,1)])
            operator_sequence_1_1.append(lambda i=i: [(step_ops[(i+1)*no_steps],0),(-1/2*dt*G*step_ops[(i+1)*no_steps+1],0)])

        calc_splitting_error(operator_sequence_1_1, step_ops, no_steps*I, order_to_run, 1, I)


    for I in range(1,5):
        no_steps = 4
        step_ops = []
        for i in range(no_steps*(I+1)): step_ops.append(1)

        #operator_sequence_1_2 = [
        #    lambda: [(1,0),(-dt*A_u,1/2),(-dt*G,0)],
        #    lambda: [(1,0),(-1/2*dt*A_u,0),(-1/2*dt*A_u*step_ops[0],0),(-1/2*dt*G,0)],
        #    lambda: [(1,0),(-dt*A_h,1/2),(-dt*F,0),(-1/2*dt*F*step_ops[1],0),(1/4*dt*dt*F*G,1)],
        #    lambda: [(step_ops[1],0),(-1/2*dt*G*step_ops[2],0)]
        #]
        operator_sequence_1_2 = [lambda: [(u_0,0)], lambda: [(u_0,0)], lambda: [(h_0,0)], lambda: [(u_0,0)]]
        for i in range(no_steps): step_ops[i] = operator_sequence_1_2[i]()[0][0]

        for i in range(0,I):
            operator_sequence_1_2.append(lambda i=i: [(step_ops[0],0),(-dt*A_u*step_ops[0],1/2),(-1/2*dt*G*step_ops[2],0),(-1/2*dt*G*step_ops[(i+1)*no_steps-2],0)])
            operator_sequence_1_2.append(lambda i=i: [(step_ops[0],0),(-1/2*dt*A_u*step_ops[0],0),(-1/2*dt*A_u*step_ops[(i+1)*no_steps],0),(-1/2*dt*G*step_ops[2],0)])
            operator_sequence_1_2.append(lambda i=i: [(step_ops[2],0),(-1/2*dt*A_h*step_ops[2],0),(-1/2*dt*A_h,1),(-1/2*dt*F*step_ops[0],0),(-1/2*dt*F*step_ops[(i+1)*no_steps+1],0),(1/4*dt*dt*F*G,1)])
            operator_sequence_1_2.append(lambda i=i: [(step_ops[(i+1)*no_steps+1],0),(-1/2*dt*G*step_ops[(i+1)*no_steps+2],0)])

        calc_splitting_error(operator_sequence_1_2, step_ops, no_steps*I, order_to_run, 2, I)


    for I in range(1,5):
        no_steps = 3
        step_ops = []
        for i in range(no_steps*(I+1)): step_ops.append(1)

        #operator_sequence_1_3_impl = [
        #    lambda: [(1,0),(-dt*A_u,1/2)],
        #    lambda: [(1,0),(-dt*A_h,1/2),(-1/2*dt*F,0),(-1/2*dt*F*step_ops[0],0),(1/2*dt*dt*F*G,1/2)],
        #    lambda: [(step_ops[0],0),(-1/2*dt*G,0),(-1/2*dt*G*step_ops[1],0)]
        #]
        operator_sequence_1_3_impl = [lambda: [(u_0,0)], lambda: [(h_0,0)], lambda: [(u_0,0)]]
        for i in range(no_steps): step_ops[i] = operator_sequence_1_3_impl[i]()[0][0]

        for i in range(0,I):
            operator_sequence_1_3_impl.append(lambda i=i: [(step_ops[0],0),(-1/2*dt*A_u*step_ops[0],0),(-1/2*dt*A_u,1)])
            operator_sequence_1_3_impl.append(lambda i=i: [(step_ops[1],0),(-1/2*dt*A_h*step_ops[1],0),(-1/2*dt*A_h,1),(-1/2*dt*F*step_ops[0],0),(-1/2*dt*F*step_ops[(i+1)*no_steps],0),(1/4*dt*dt*F*G*step_ops[1],0),(1/4*dt*dt*F*G,1)])
            operator_sequence_1_3_impl.append(lambda i=i: [(step_ops[(i+1)*no_steps],0),(-1/2*dt*G*step_ops[(i+1)*no_steps+1],0),(-1/2*dt*G*step_ops[1],0)])

        calc_splitting_error(operator_sequence_1_3_impl, step_ops, no_steps*I, order_to_run, 3, I)


    for I in range(1,5):
        no_steps = 3
        step_ops = []
        for i in range(no_steps*(I+1)): step_ops.append(1)

        #operator_sequence_1_3_expl = [
        #    lambda: [(1,0),(-dt*A_u,1/2)],
        #    lambda: [(1,0),(-dt*A_h,0),(-1/2*dt*F,0),(-1/2*dt*F*step_ops[0],0),(1/2*dt*dt*F*G,1/2)],
        #    lambda: [(step_ops[0],0),(-1/2*dt*G,0),(-1/2*dt*G*step_ops[1],0)]
        #]
        operator_sequence_1_3_expl = [lambda: [(u_0,0)], lambda: [(h_0,0)], lambda: [(u_0,0)]]
        for i in range(no_steps): step_ops[i] = operator_sequence_1_3_expl[i]()[0][0]

        for i in range(0,I):
            operator_sequence_1_3_expl.append(lambda i=i: [(step_ops[0],0),(-1/2*dt*A_u*step_ops[0],0),(-1/2*dt*A_u,1)])
            operator_sequence_1_3_expl.append(lambda i=i: [(step_ops[1],0),(-1/2*dt*A_h*step_ops[1],0),(-1/2*dt*A_h*step_ops[(i+1)*no_steps-2],0),(-1/2*dt*F*step_ops[0],0),(-1/2*dt*F*step_ops[(i+1)*no_steps],0),(1/4*dt*dt*F*G*step_ops[1],0),(1/4*dt*dt*F*G,1)])
            operator_sequence_1_3_expl.append(lambda i=i: [(step_ops[(i+1)*no_steps],0),(-1/2*dt*G*step_ops[(i+1)*no_steps+1],0),(-1/2*dt*G*step_ops[1],0)])


        calc_splitting_error(operator_sequence_1_3_expl, step_ops, no_steps*I, order_to_run, 4, I)
