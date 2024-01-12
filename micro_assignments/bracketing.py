import numpy as np

def f(x):
    return (x**3 + x - 10) #root = 2

def bisection_iterative(f, a,b,tol):
    m = (a + b)/2 #step two: midpoint
    
    #step three: iterative 
    if np.abs(f(m)) < tol:
        return m
    elif np.sign(f(a)) == np.sign(f(m)):
        return bisection_iterative(f, m, b, tol)
    elif np.sign(f(b)) == np.sign(f(m)):
        return bisection_iterative(f, a, m, tol)

def bisection_error(a,b,tol):
    c = a

    #step four: error estimate
    cur_n = b
    prev_n = a
    while ((cur_n-prev_n) >= tol):
        c = (a+b)/2 #step two: midpoint
        prev_n = c
        #step three: iterative
        if (f(c) == 0.0):
            break
        if (f(c)*f(a) < 0):
            b = c
            cur_n = (a+b)/2
        else:
            a = c   
            cur_n = (a+b)/2   
    return c 

print("The root is: {}".format(bisection_iterative(f, 0, 5, 0.01))) #step one: choose brackets
print("The root is: {}".format(bisection_error(0, 5, 0.0001))) 