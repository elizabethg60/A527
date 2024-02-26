import numpy as np

def trapezoid(f, a, b, n):
# returns integration value using the composite trapezoidal rule
    h = (b - a) / n

    integral = (f(a) + f(b)) / 2
    for i in range(1, n):
        integral = integral + f(a + i*h)
    
    return integral * h 

def monte_carlo(f, a, b, n): 
# returns integration value using the Monte Carlo method
    x = np.linspace(a, b, n)
    mean_f = np.sum(f(x)) / n

    return (b - a) * mean_f