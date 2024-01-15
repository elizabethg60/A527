import numpy as np

def potential(m1, m2, d):
    m = m1 + m2
    omega_squared = m / (d**3)
    x1 = -(m2/m)*d
    x2 = (m1/m)*d
    r1_vector = np.array([x1, 0])
    r2_vector = np.array([x2, 0])
    r_vector = r2_vector-r1_vector 
    r = np.sqrt(r_vector[0]**2 + r_vector[1]**2)

    return -m1/((np.abs(list(r_vector-r1_vector))[0])) - m2/((np.abs(list(r_vector-r2_vector))[0])) - 0.5*omega_squared*r**2

def acceleration(m1, m2, d):
    m = m1 + m2
    omega_squared = m / (d**3)
    x1 = -(m2/m)*d
    x2 = (m1/m)*d
    r1_vector = np.array([x1, 0])
    r2_vector = np.array([x2, 0])
    r_vector = r2_vector-r1_vector
    r = np.sqrt(r_vector[0]**2 + r_vector[1]**2)

    return np.nan_to_num(-(m1*(r_vector-r1_vector))/(np.abs(r_vector-r1_vector)**3) - (m2*(r_vector-r2_vector))/(np.abs(r_vector-r2_vector)**3) + omega_squared*r_vector)

def potential_grid(m1, m2, d, orbit1, orbit2):
    m = m1 + m2
    omega_squared = m / (d**3)
    r1_vector = np.array([orbit1, orbit1])
    r2_vector = np.array([orbit2, orbit2])
    r_vector = r2_vector-r1_vector 
    r = np.sqrt(r_vector[0]**2 + r_vector[1]**2)

    return -m1/((np.abs(list(r_vector-r1_vector))[0])) - m2/((np.abs(list(r_vector-r2_vector))[0])) - 0.5*omega_squared*r**2

def acceleration_grid(m1, m2, d, orbit1, orbit2):
    m = m1 + m2
    omega_squared = m / (d**3)
    r1_vector = np.array([orbit1, orbit1])
    r2_vector = np.array([orbit2, orbit2])
    r_vector = r2_vector-r1_vector
    r = np.sqrt(r_vector[0]**2 + r_vector[1]**2)

    return np.nan_to_num(-(m1*(r_vector-r1_vector))/(np.abs(r_vector-r1_vector)**3) - (m2*(r_vector-r2_vector))/(np.abs(r_vector-r2_vector)**3) + omega_squared*r_vector)
