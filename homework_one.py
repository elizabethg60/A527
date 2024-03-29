from A527_package import potential, acceleration, bisection_xaxis, bisection_yaxis
import matplotlib.pyplot as plt
import numpy as np

"""
Consider two bodies of mass m1 and m2 separated by a distance d traveling in
circular orbits around their mutual center of mass. Take the angular momentum vector of
the system to be pointing in the +z direction. In a frame that co-rotates with the orbital
motion there are five locations (called the Lagrange points) where the effective acceleration
felt by a test particle vanishes. The acceleration arises from the gravity due to m1 and m2
plus the rotation of the system as a whole. If the masses are at points x1 < x2 on the x axis
(so d = x2 − x1), then by convention L3 < x1 < L1 < x2 < L2. The L4 and L5 points lie
off the x axis and form equilateral triangles with the points x1 and x2. Conventionally, L4
is taken to be in the +y direction, L5 in the −y direction.
"""

"""
(1). [30 points] Write a program to explore this system by computing the effective
gravity (a vector) and potential (a scalar) at specific grid points in the xy-plane. Use units
such that the gravitational constant G ≡ 1. The program should take as input the mass of
both bodies and their separation. You may “hardwire” the grid dimensions into your code
if you wish. The output should be the potential and x and y components of the acceleration
at each grid point.
"""

#hardwire grid dimensions
grid_x = np.linspace(-1.5, 1.5, 500)
grid_y = np.linspace(-1.5, 1.5, 500)

grid_x_small = np.linspace(-1.5, 1.5, 10)
grid_y_small = np.linspace(-1.5, 1.5, 10)

#function for potential and acceleration (ie gravity) found under A527_package in JWST.py

"""
(2). [20 points] Use your favorite plotting program to plot vectors (for the effective
acceleration) and contours (for the effective potential) for the cases where m1 = 3, m2 = 1,
d = 1 and m1 = 100, m2 = 1, d = 1.

(3). [20 points] Use a simple root solver to determine the locations of the Lagrange points
for both these cases, using your a priori knowledge of their approximate locations (i.e., first do a search along the x axis for the L1, L2, and L3 points, then along the x = (x1 + x2)/2
axis for the L4 and L5 points. Watch out for singularities!...).
"""
tol = 0.001
#case one: m1 = 3, m2 = 1, d = 1
m1 = 3
m2 = 1
d = 1
m = m1 + m2
x1 = -(m2/m)*d
x2 = (m1/m)*d
x = (x1 + x2)/2

X, Y = np.meshgrid(grid_x, grid_y)
X_small, Y_small = np.meshgrid(grid_x_small, grid_y_small)

Z = []
gravity_x = []
gravity_y = []
for i in range(0, len(X)):
    inner_Z = []
    for j in range(0,len(X[i])):
        inner_Z.append(potential(m1, m2, d, X[i][j], Y[i][j]))
    Z.append(inner_Z)

for i in range(0, len(X_small)):
    inner_gravity_x = []
    inner_gravity_y = []
    for j in range(0,len(X_small[i])):
        inner_gravity_x.append(list(acceleration(m1, m2, d, X_small[i][j], Y_small[i][j]))[0])
        inner_gravity_y.append(list(acceleration(m1, m2, d, X_small[i][j], Y_small[i][j]))[1])
    gravity_x.append(inner_gravity_x)
    gravity_y.append(inner_gravity_y)

L3_x = bisection_xaxis(acceleration, m1, m2, d, -2, x1 + 0.001, tol) 
L1_x = bisection_xaxis(acceleration, m1, m2, d, x1 + 0.001, x2 + 0.001, tol)
L2_x = bisection_xaxis(acceleration, m1, m2, d, x2 + 0.001, 2, tol)

L4_y = bisection_yaxis(acceleration, m1, m2, d, 0, 2, tol, x) 
L5_y = bisection_yaxis(acceleration, m1, m2, d, -2, 0, tol, x)

# plot
fig, ax = plt.subplots()
min_level = np.quantile(Z, 0.05)
max_level = np.quantile(Z, 0.95)
levels = np.linspace(min_level, max_level, 15)
ax.contour(X, Y, Z, levels=levels) #potential
ax.quiver(X_small, Y_small, gravity_x, gravity_y, color="C0", angles='xy', scale_units='xy', scale=4, width=.005) #gravity
ax.scatter([-(m2/m)*d, (m1/m)*d], [0,0], color = 'r', label = "bodies")
ax.scatter([L1_x,L2_x,L3_x,x, x], [0,0,0,L4_y, L5_y], color = 'k')
L_labels = ["L1", "L2", "L3", "L4", "L5"]
for i, txt in enumerate(L_labels):
    ax.annotate(txt, ([L1_x,L2_x,L3_x,x, x][i], [0,0,0,L4_y, L5_y][i]))
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.savefig("Figures/homework_one/case_one.png")
plt.show()

#case two: m1 = 100, m2 = 1, d = 1
m1 = 100
m2 = 1
d = 1
m = m1 + m2
x1 = -(m2/m)*d
x2 = (m1/m)*d
x = (x1 + x2)/2

Z = []
gravity_x = []
gravity_y = []
for i in range(0, len(X)):
    inner_Z = []
    for j in range(0,len(X[i])):
        inner_Z.append(potential(m1, m2, d, X[i][j], Y[i][j]))
    Z.append(inner_Z)

for i in range(0, len(X_small)):
    inner_gravity_x = []
    inner_gravity_y = []
    for j in range(0,len(X_small[i])):
        inner_gravity_x.append(list(acceleration(m1, m2, d, X_small[i][j], Y_small[i][j]))[0])
        inner_gravity_y.append(list(acceleration(m1, m2, d, X_small[i][j], Y_small[i][j]))[1])
    gravity_x.append(inner_gravity_x)
    gravity_y.append(inner_gravity_y)

L3_x = bisection_xaxis(acceleration, m1, m2, d, -2, x1 + 0.001, tol) 
L1_x = bisection_xaxis(acceleration, m1, m2, d, x1 + 0.001, x2 + 0.001, tol)
L2_x = bisection_xaxis(acceleration, m1, m2, d, x2 + 0.001, 2, tol)

L4_y = bisection_yaxis(acceleration, m1, m2, d, 0, 2, tol, x) 
L5_y = bisection_yaxis(acceleration, m1, m2, d, -2, 0, tol, x)

# plot
fig, ax = plt.subplots()
min_level = np.quantile(Z, 0.05)
max_level = np.quantile(Z, 0.95)
levels = np.linspace(min_level, max_level, 15)
ax.contour(X, Y, Z, levels=levels) 
ax.quiver(X_small, Y_small, gravity_x, gravity_y, color="C0", angles='xy', scale_units='xy', scale=4, width=.005)
ax.scatter([-(m2/m)*d, (m1/m)*d], [0,0], color = 'r', label = "bodies")
ax.scatter([L1_x,L2_x,L3_x,x, x], [0,0,0,L4_y, L5_y], color = 'k')
L_labels = ["L1", "L2", "L3", "L4", "L5"]
for i, txt in enumerate(L_labels):
    ax.annotate(txt, ([L1_x,L2_x,L3_x,x, x][i], [0,0,0,L4_y, L5_y][i]))
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.savefig("Figures/homework_one/case_two.png")
plt.show()

"""
(4) [30 points] The James Webb Space Telescope was successfully launched on 12/24/2021.
It is the largest and the most complex telescope ever launched into space. It will observe
primarily the infrared light from faint and very distant objects. But all objects, including
telescopes, also emit infrared light. To avoid swamping the very faint astronomical sig-
nals with radiation from the telescope, the telescope and its instruments must be very cold.
Therefore, JWST has a large shield that blocks the light from the Sun, Earth, and Moon. To
have this work, JWST must be in an orbit where all three of these objects are in about the
same direction. Please calculate and plot the ideal location and orbit for JWST.
"""