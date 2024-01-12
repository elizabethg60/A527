import matplotlib.pyplot as plt
import numpy as np

def gravitational_potential(M,m,x,y,z):
    G = 6.67*10**(-8) #dyne cm2 g-2
    r = np.sqrt(x**2+y**2+z**2)
    return  -(G*M*m)/r

M = 2*10**(33) #g
m = 5.97*10**(27)
x = np.linspace(-15,15,100)
y = np.linspace(-15,15,100)
z = np.linspace(-5,5,100)

X, Y = np.meshgrid(x, y)
Z = gravitational_potential(M,m,X,Y,z)
levels = np.linspace(np.min(Z), np.max(Z), 7)


# plot
fig, ax = plt.subplots()

ax.contour(X, Y, Z, levels=levels)

U = X + Y
V = Y - X
ax.quiver(X, Y, U, V, color="C0", angles='xy', scale_units='xy', scale=4, width=.005)

ax.set(xlim=(-2, 2), ylim=(-2, 2))
plt.show()