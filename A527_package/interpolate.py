import numpy as np

def lagrange_polynomial(x_arr, i, x):
# returns lagrange basis polynomial for arbitrary degree
    total = 1
    for j, x_j in enumerate(x_arr):
        if j != i:
            total = total * (x - x_j) / (x_arr[i] - x_j)
    return total

def lagrange_interpolation(x_arr, y_arr, x):
# returns lagrange interpolation runge value for a given x 
    total = 0.0
    for i in range(0, len(x_arr)):
        total = total + (y_arr[i] * lagrange_polynomial(x_arr, i, x))
    return total

def cubic_spline_interpolation(n, x, y): 
# returns cubic spline interpolation runge values for given set of x and arbitrary interval n
	h = [] 
	for i in range(n-1):
		h.append(x[i+1]-x[i])

	p2prime = np.zeros(n) 

	a, b, c, r = np.zeros(n-1), np.zeros(n-1), np.zeros(n-1), np.zeros(n-1) 
	for i in range(n-2): 
		r[i] = 6*(((y[i+2] - y[i+1])/h[i+1]) - ((y[i+1] - y[i])/h[i]))
		b[i] = 2*(h[i] + h[i+1])
		if (i < n-2): c[i] = h[i+1] 
		if (i > 0): a[i] = h[i] 

	beta = b
	rho = r 
	for j in range(1, n-1):
		beta[j] = b[j] - (a[j] * c[j-1])/beta[j-1]
		rho[j] = r[j] - (a[j] * rho[j-1])/beta[j-1]

	p2prime[n-2] = rho[n-3]/beta[n-3] 
	for j in range(1, n-2):
		p2prime[n-j-2] = (rho[n-3-j] - c[n-3-j]*p2prime[n-j])/beta[n-3-j]

	interpolation = np.zeros(n) 
	for j in range(n-1):
		cond = np.logical_and(x >= x[j], x <= x[j+1]) 
		interpolation[cond] = (p2prime[j+1]-p2prime[j])/(6*h[j]) * (x[cond]-x[j])**3 + p2prime[j]/2 * (x[cond]-x[j])**2 + ((y[j+1]-y[j])/h[j] - h[j]*p2prime[j+1]/6 - h[j]*p2prime[j]/3)*(x[cond]-x[j]) + y[j] 

	return interpolation