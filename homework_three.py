import numpy as np
import matplotlib.pyplot as plt
from A527_package import lagrange_polynomial, lagrange_interpolation, cubic_spline_interpolation, trapezoid, monte_carlo

"""
1. Data Interpolation.
Plot the Runge function in the range of x=[-1,1].
"""

def runge_fct(x):
# returns runge value for a given x
    return 1/(25*x**2+1)

#hardwire range of x=[-1,1]
x = np.linspace(-1, 1, 200)
#corresponding runge function
y = runge_fct(x)

#plot 
fig1, ax1 = plt.subplots()
ax1.plot(x, y, color = 'k',linewidth=2, label = 'Runge Curve')
ax1.set_xlabel('x')
ax1.set_ylabel('Runge Function')
ax1.legend()
plt.savefig("Figures/homework_three/fit_a.pdf")
plt.show()

"""
Write a program to generate the Lagrange interpolation polynomials
of arbitrary degree n based on n + 1 data points. Over-plot the Lagrange
polynomials with n=6, n=8, and n=10, on top of the Runge curve from (1).
"""

#Lagrange interpolation with n=6
n = 6
x_6 = np.linspace(-1, 1, n + 1)
y_6 = runge_fct(x_6)
lagrange_6 = []
for i in x:
    lagrange_6.append(lagrange_interpolation(x_6, y_6, i))

#Lagrange interpolation with n=8
n = 8
x_8 = np.linspace(-1, 1, n + 1)
y_8 = runge_fct(x_8)
lagrange_8 = []
for i in x:
    lagrange_8.append(lagrange_interpolation(x_8, y_8, i))

#Lagrange interpolation with n=10
n = 10
x_10 = np.linspace(-1, 1, n + 1)
y_10 = runge_fct(x_10)
lagrange_10 = []
for i in x: 
    lagrange_10.append(lagrange_interpolation(x_10, y_10, i))

#plot 
fig1, ax1 = plt.subplots()
ax1.plot(x, y, color = 'k',linewidth=2, label = 'Runge Curve')
plt.plot(x, lagrange_6, label=f'Lagrange n=6', color='r')
plt.plot(x, lagrange_8, label=f'Lagrange n=8', color='b')
plt.plot(x, lagrange_10, label=f'Lagrange n=10', color='g')
ax1.set_xlabel('x')
ax1.set_ylabel('Runge Function')
ax1.legend()
plt.savefig("Figures/homework_three/fit_b.pdf")
plt.show()

"""
Write a program to implement cubic natural spline interpolation on
10 intervals from the Rung function, and over-plot the cubic spline curve on
top of the previous curves from (1) and (2). 
"""

#cubic natural spline interpolation with 10 intervals 
n = 10
x_cubic = np.linspace(-1, 1, n)
y_cubic_runge = runge_fct(x_cubic)
y_cubic = cubic_spline_interpolation(n, x_cubic, y_cubic_runge)

#plot 
fig1, ax1 = plt.subplots()
ax1.plot(x, y, color = 'k',linewidth=2, label = 'Runge Curve')
plt.plot(x, lagrange_6, label=f'Lagrange n=6', color='r')
plt.plot(x, lagrange_8, label=f'Lagrange n=8', color='b')
plt.plot(x, lagrange_10, label=f'Lagrange n=10', color='g')
plt.plot(x_cubic, y_cubic, label='Cubic Spline', color='y')
ax1.set_xlabel('x')
ax1.set_ylabel('Runge Function')
ax1.legend()
plt.savefig("Figures/homework_three/fit_c.pdf")
plt.show()

"""
2. Numerical Integration.
Write a program to implement the Composite Trapezoidal Rule for integration
Write a program to implement the Monte Carlo method for integration.
"""

#both functions can be found under A527_package in integration.py 

"""
Now, apply these two programs to calculate the following
integration, and use the comparison of the results from these algorithms with
that from analytical solution to estimate the errors. 
"""

def integrate_fct(x):
    return np.exp((-(x-1)**2)/2)/np.sqrt(2*np.pi)

#integral bounds
a = -100
b = 100
n = 100

print("analytical solution: 1")
t_soln = trapezoid(integrate_fct, a, b, 100)
t_err = np.abs(t_soln - 1) * 100
print("trapezoid solution: {} with error of {}%".format(t_soln, round(t_err, 3)))
mc_soln = monte_carlo(integrate_fct, a, b, 100)
mc_err = np.abs(mc_soln - 1) * 100
print("monte carlo solution: {} with error of {}%".format(mc_soln, round(mc_err,3)))
