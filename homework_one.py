from A527_package import potential, acceleration, potential_grid, acceleration_grid
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

#function potential and acceleration (fixed grid dimensions as specified in hw hints) found under A527_package in JWST.py
#function potential_grid and acceleration_grid (input circular orbital grid dimensions) found under A527_package in JWST.py

"""
(2). [20 points] Use your favorite plotting program to plot vectors (for the effective
acceleration) and contours (for the effective potential) for the cases where m1 = 3, m2 = 1,
d = 1 and m1 = 100, m2 = 1, d = 1.
"""
#case one: m1 = 3, m2 = 1, d = 1
m1 = 3
m2 = 1
d = 1
m = m1 + m2
x1 = -(m2/m)*d
x2 = (m1/m)*d
orbit1x = np.linspace(-x1, x1, 50)
orbit2x = np.linspace(-x2, x2, 50)

def y_circle_pos(x, radius):
    return np.sqrt(radius**2 - x**2)

def y_circle_neg(x, radius):
    return -np.sqrt(radius**2 - x**2)

orbit1y_pos = y_circle_pos(orbit1x, np.abs(x1))
orbit2y_pos = y_circle_pos(orbit2x, np.abs(x2))
orbit1y_neg = y_circle_neg(orbit1x, np.abs(x1))
orbit2y_neg = y_circle_neg(orbit2x, np.abs(x2))

orbit1y = list(orbit1y_pos) + list(orbit1y_neg)
orbit2y = list(orbit2y_pos) + list(orbit2y_neg)
orbit1x = list(orbit1x) + list(orbit1x)
orbit2x = list(orbit2x) + list(orbit2x)

# print(potential(m1, m2, d))
# print(acceleration(m1, m2, d))

#potential_array = []
acceleration_array = []
import math
for i in range(0, len(orbit1x)):
    #potential_array.append(potential_grid(m1, m2, d, orbit1x[i], orbit1y[i], orbit2x[i], orbit2y[i]))
    acceleration_array.append(acceleration_grid(m1, m2, d, orbit1x[i], orbit1y[i], orbit2x[i], orbit2y[i]))
    print(math.dist([orbit1x[i], orbit1y[i]], [orbit2x[i], orbit2y[i]])) 

# print(acceleration_array)

#plot object one 
X1, Y1 = np.meshgrid(orbit1x, orbit1y)
X2, Y2 = np.meshgrid(orbit2x, orbit2y)
Z = potential_grid(m1, m2, d, X1, Y1, X2, Y2)
levels = np.linspace(np.min(Z), np.max(Z), 7)

#plot
fig, ax = plt.subplots()
ax.scatter(orbit1x, orbit1y, color = 'r')
ax.scatter(orbit2x, orbit2y, color = 'r')
ax.contour(X1, Y1, Z, levels=levels)
ax.contour(X2, Y2, Z, levels=levels)

ax.quiver(orbit1x, orbit1y, list(list(zip(*acceleration_array)))[0], list(list(zip(*acceleration_array)))[1], color="C0", angles='xy', scale_units='xy', scale=4, width=.005)
ax.quiver(orbit2x, orbit2y, list(list(zip(*acceleration_array)))[0], list(list(zip(*acceleration_array)))[1], color="C0", angles='xy', scale_units='xy', scale=4, width=.005)
plt.show()

#Wednesday: debug 1 + 2
#confirm with cindy: geometry, how to input grid location, distance not always 1 along circle
#confirm debuging complete for 1 + 2 then do case #2

"""
(3). [20 points] Use a simple root solver to determine the locations of the Lagrange points
for both these cases, using your a priori knowledge of their approximate locations (i.e., first do a search along the x axis for the L1, L2, and L3 points, then along the x = (x1 + x2)/2
axis for the L4 and L5 points. Watch out for singularities!...).
"""

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