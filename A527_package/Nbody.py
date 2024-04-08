import numpy as np

def getAcc(pos, mass, G, N):
# returns acceleration given positions and masses
	A = np.zeros_like(pos)
	for i in range(N):
		for j in range(N):
			if i != j:
				r = pos[j] - pos[i]
				A[i] += G * mass[j] * r / np.linalg.norm(r)**3
	return A

def getEnergy(pos, vel, mass, G, N):
# returns KE and PE given positions, velocities, and masses
	# Kinetic Energy:
	KE = 0.5 * np.sum(np.sum( mass * vel**2 ))

    # Potential Energy:
	PE = np.zeros_like(mass)
	for i in range(N):
		for j in range(N):
			if i != j:
				r = pos[j] - pos[i]
				PE[i] += G * mass[i] * mass[j] / np.linalg.norm(r)	
	return KE, np.sum(PE)

def leapFrog(N, Nt, dt, pos, vel, acc, mass, G, KE_save, PE_save, pos_save):
# performs leap frog algorithm and updates KE, PE, and pos 
    for i in range(Nt):
        vel += acc * dt/2.0
        # drift
        pos += vel * dt	
        # update accelerations
        acc = getAcc(pos, mass, G, N)
        # (1/2) kick
        vel += acc * dt/2.0
        # get energy of system
        KE, PE  = getEnergy(pos, vel, mass, G, N)
            
        # save energies, positions for plotting trail
        pos_save[i+1] = pos
        KE_save[i+1] = KE
        PE_save[i+1] = PE 

def RK4(N, Nt, dt, pos, vel, mass, G, KE_save, PE_save, pos_save):
# performs RK4 algorithm and updates KE, PE, and pos
    for i in range(Nt):
        k1vel = dt * getAcc(pos, mass, G, N)
        k1pos = dt * vel
        
        k2vel = dt * getAcc(pos + 0.5 * k1pos, mass, G, N)
        k2pos = dt * (vel + 0.5 * k1vel)
        
        k3vel = dt * getAcc(pos + 0.5 * k2pos, mass, G, N)
        k3pos = dt * (vel + 0.5 * k2vel)
        
        k4vel = dt * getAcc(pos + k3pos, mass, G, N)
        k4pos = dt * (vel + k3vel)
        
        pos += (k1pos + 2*k2pos + 2*k3pos + k4pos) / 6
        vel += (k1vel + 2*k2vel + 2*k3vel + k4vel) / 6

        KE, PE  = getEnergy(pos, vel, mass, G, N)
        
        # save energies, positions for plotting trail
        pos_save[i+1] = pos
        KE_save[i+1] = KE
        PE_save[i+1] = PE 