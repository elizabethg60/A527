import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt 
from A527_package import levenberg_marquardt as LM

"""
Write a least-squares fitting code that implements the Levenberg-
Marquardt algorithm
Use the above least-squares fitting code to fit the data hw2 fitting.dat
on CANVAS with two functions: a Lorentzian as in Equation (1), and a
Gaussian as in Equation (2)
Plot the data and the fitting curves, be sure to include the
error-bars of the data. Which function fits the data better and why?
"""

#set whether to use Lorentzian or Gaussian function 
func = "L"

#initial guess for parameters 
p_init = np.array([[50,50]]).T 

#read in data
with open('hw2_fitting.dat') as f:
    x = np.array([float(line.split()[0]) for line in f])
with open('hw2_fitting.dat') as f:
    y = np.array([float(line.split()[1]) for line in f])
with open('hw2_fitting.dat') as f:
    sigma = np.array([float(line.split()[2]) for line in f])

#L-M least squares analysis
"""
p_fit   : least-squares optimal estimate of the parameter values
Chi_sq  : reduced Chi squared error criteria - should be close to 1
sigma_p : asymptotic standard error of the parameters
sigma_y : asymptotic standard error of the curve-fit
corr    : correlation matrix of the parameters
R_sq    : R-squared cofficient of multiple determination  
cvg_hst : convergence history (col 1: function calls, col 2: reduced chi-sq,
              col 3 through n: parameter values). Row number corresponds to
              iteration number.
"""
p_fit,Chi_sq,sigma_p,sigma_y,corr,R_sq,cvg_hst = LM.lm(p_init,x,y,sigma,func)
    
#plot results 
# extract parameters data
p_hst  = cvg_hst[:,3:]
p_fit  = p_hst[-1,:]
y_fit = LM.lm_func(x,np.array([p_fit]).T, func)     
    
# define colors and markers used for plotting
n = len(p_fit)
colors = pl.cm.ocean(np.linspace(0,.75,n))
markers = ['o','s']    
    
# create plot of raw data and fitted curve
fig1, ax1 = plt.subplots()
ax1.errorbar(x,y, yerr = sigma, fmt = 'o',color='black',label='Raw data')
ax1.plot(x,y_fit,'r--',label='Fitted curve',linewidth=2)
ax1.set_xlabel('frequency')
ax1.set_ylabel('line strength')
ax1.set_title('Data fitting')
ax1.legend()
plt.savefig("Figures/homework_two/fit_{}.pdf".format(func))
    
# create plot showing convergence of parameters
fig2, ax2 = plt.subplots()
for i in range(n):
    ax2.plot(cvg_hst[:,0],p_hst[:,i]/p_hst[0,i], color=colors[i],marker=markers[i], linestyle='-',label='p'+'${_%i}$'%(i+1))
ax2.set_xlabel('Function calls')
ax2.set_ylabel('Values (norm.)')
ax2.set_title('Convergence of parameters') 
ax2.legend()    
plt.savefig("Figures/homework_two/parameters_{}.pdf".format(func))

# create plot showing evolution of reduced chi-sq
fig3, ax3 = plt.subplots()
ax3.plot(cvg_hst[:,0],cvg_hst[:,1], color='k', marker='o', linestyle='-')
ax3.set_xlabel('Function calls')
ax3.set_ylabel('reduced chi-sq')
ax3.set_title('Evolution of reduced chi-sq') 
plt.savefig("Figures/homework_two/reducedchi_{}.pdf".format(func))

# create plot showing evolution of lambda
fig3, ax3 = plt.subplots()
ax3.plot(cvg_hst[:,0],cvg_hst[:,2], color='k', marker='o', linestyle='-')
ax3.set_xlabel('Function calls')
ax3.set_ylabel('lambda')
ax3.set_title('Evolution of lambda') 
plt.savefig("Figures/homework_two/lambda_{}.pdf".format(func))

plt.show()