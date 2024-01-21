import numpy as np

def potential(m1, m2, d, x, y):
#funtion that computes potential [scalar] of stationary point (x,y) for 2 body system with mass m1 and m2 and distance d
    m = m1 + m2
    omega_squared = m / (d**3)
    x1 = -(m2/m)*d
    x2 = (m1/m)*d
    r1_vector = np.array([x1, 0])
    r2_vector = np.array([x2, 0])
    r_vector = np.array([x, y])
    r = np.linalg.norm(r_vector)

    return -m1/((np.linalg.norm(list(r_vector-r1_vector)))) - m2/((np.linalg.norm(list(r_vector-r2_vector)))) - 0.5*omega_squared*r**2

def acceleration(m1, m2, d, x, y):
#funtion that computes acceleration (ie gravity) [vector] of stationary point (x,y) for 2 body system with mass m1 and m2 and distance d
    m = m1 + m2
    omega_squared = m / (d**3)
    x1 = -(m2/m)*d
    x2 = (m1/m)*d
    r1_vector = np.array([x1, 0])
    r2_vector = np.array([x2, 0])
    r_vector = np.array([x, y])
    r = np.linalg.norm(r_vector)

    return (-m1/((np.linalg.norm(list(r_vector-r1_vector)))**3))*(r_vector-r1_vector) - (m2/((np.linalg.norm(list(r_vector-r2_vector)))**3))*(r_vector-r2_vector) + omega_squared*r_vector

def bisection_xaxis(f, m1, m2, d, a, b, tol):
#search along the x axis for the L1, L2, and L3 points
    c = (a+b)/2.0
    y = 0
    while (b-a)/2.0 > tol:
        if sum(list(f(m1, m2, d, c, y))) == 0:
            return c
        elif sum(list(f(m1, m2, d, a, y)))*sum(list(f(m1, m2, d, c, y))) < 0:
            b = c
        else :
            a = c
        c = (a+b)/2.0
    return c

def bisection_yaxis(f, m1, m2, d, a, b, tol, x):
#search along the x = (x1 + x2)/2 line for the L4 and L5 points
    c = (a+b)/2.0
    while (b-a)/2.0 > tol:
        if sum(list(f(m1, m2, d, x, c))) == 0:
            return c
        elif sum(list(f(m1, m2, d, x, a)))*sum(list(f(m1, m2, d, x, c))) < 0:
            b = c
        else :
            a = c
        c = (a+b)/2.0
    return c