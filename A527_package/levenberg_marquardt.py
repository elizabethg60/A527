import numpy as np

def lm_func(t,p,func):
    #Lorentzian 
    if func == "L":
        return p[0,0]/(np.pi*((t-p[1,0])**2+p[0,0]**2))

    #Gaussian
    if func == "G":
        return (1/p[0,0])*(np.sqrt(np.log(2)/np.pi))*np.exp((-np.log(2)*(t-p[1,0])**2)/p[0,0]**2)

def lm_FD_J(t,p,y,dp,func):
    """
    Computes partial derivates (Jacobian) dy/dp via finite differences.

    Parameters
    ----------
    t  :     independent variables used as arg to lm_func (m x 1) 
    p  :     current parameter values (n x 1)
    y  :     func(t,p,c) initialised by user before each call to lm_FD_J (m x 1)
    dp :     fractional increment of p for numerical derivatives
                - dp(j)>0 central differences calculated
                - dp(j)<0 one sided differences calculated
                - dp(j)=0 sets corresponding partials to zero; i.e. holds p(j) fixed
    Returns
    -------
    J :      Jacobian Matrix (n x m)
    """

    global func_calls
    
    # number of data points
    m = len(y)
    # number of parameters
    n = len(p)

    # initialize Jacobian to Zero
    ps=p
    J=np.zeros((m,n)) 
    del_=np.zeros((n,1))
    
    for j in range(n):
        # parameter perturbation
        del_[j,0] = dp[j,0] * (1+abs(p[j,0]))
        # perturb parameter p(j)
        p[j,0]   = ps[j,0] + del_[j,0]

        J[:,j] = (lm_func(t,p,func)-y)/del_[j,0]
        func_calls = func_calls + 1

    return J
    
def lm_Broyden_J(p_old,y_old,J,p,y):
    """
    Carry out a rank-1 update to the Jacobian matrix using Broyden's equation.

    Parameters
    ----------
    p_old :     previous set of parameters (n x 1)
    y_old :     model evaluation at previous set of parameters, y_hat(t,p_old) (m x 1)
    J     :     current version of the Jacobian matrix (m x n)
    p     :     current set of parameters (n x 1)
    y     :     model evaluation at current  set of parameters, y_hat(t,p) (m x 1)

    Returns
    -------
    J     :     rank-1 update to Jacobian Matrix J(i,j)=dy(i)/dp(j) (m x n)

    """
    
    h = p - p_old
    
    a = np.dot((np.array([y - y_old]).T - np.dot(J,h)),h.T)
    b = np.dot(h.T,h)

    # Broyden rank-1 update eq'n
    J = J + a/b

    return J

def lm_matx(t,p_old,y_old,dX2,J,p,y_dat,weight,dp,func):
    """
    Evaluate the linearized fitting matrix, JtWJ, and vector JtWdy, and 
    calculate the Chi-squared error function, Chi_sq used by Levenberg-Marquardt 
    algorithm (lm).
    
    Parameters
    ----------
    t      :     independent variables used as arg to lm_func (m x 1)
    p_old  :     previous parameter values (n x 1)
    y_old  :     previous model ... y_old = y_hat(t,p_old) (m x 1)
    dX2    :     previous change in Chi-squared criteria (1 x 1)
    J      :     Jacobian of model, y_hat, with respect to parameters, p (m x n)
    p      :     current parameter values (n x 1)
    y_dat  :     data to be fit by func(t,p,c) (m x 1)
    weight :     the weighting vector for least squares fit inverse of 
                 the squared standard measurement errors
    dp     :     fractional increment of 'p' for numerical derivatives
                  - dp(j)>0 central differences calculated
                  - dp(j)<0 one sided differences calculated
                  - dp(j)=0 sets corresponding partials to zero; i.e. holds p(j) fixed
    Returns
    -------
    JtWJ   :     linearized Hessian matrix (inverse of covariance matrix) (n x n)
    JtWdy  :     linearized fitting vector (n x m)
    Chi_sq :     Chi-squared criteria: weighted sum of the squared residuals WSSR
    y_hat  :     model evaluated with parameters 'p' (m x 1)
    J :          Jacobian of model, y_hat, with respect to parameters, p (m x n)
    """
    
    global iteration
    
    # number of parameters
    Npar   = len(p)

    # evaluate model using parameters 'p'
    y_hat = lm_func(t,p,func)

    if not np.remainder(iteration,2*Npar) or dX2 > 0:
        # finite difference
        J = lm_FD_J(t,p,y_hat,dp,func)
    else:
        # rank-1 update
        J = lm_Broyden_J(p_old,y_old,J,p,y_hat)

    # residual error between model and data
    delta_y = np.array([y_dat - y_hat]).T
    
    # Chi-squared error criteria
    Chi_sq = np.dot(delta_y.T, ( delta_y * weight ))

    JtWJ  = np.dot(J.T, ( J * ( weight * np.ones((1,Npar)) ) ))
    
    JtWdy = np.dot(J.T, ( weight * delta_y ))  
    
    return JtWJ,JtWdy,Chi_sq,y_hat,J


def lm(p,t,y_dat,sigma,func):  
    """
    Levenberg Marquardt curve-fitting: minimize sum of weighted squared residuals

    Parameters
    ----------
    p : initial guess of parameter values (n x 1)
    t : independent variables (used as arg to lm_func) (m x 1)
    y_dat : data to be fit by func(t,p) (m x 1)
    Returns
    -------
    p       : least-squares optimal estimate of the parameter values
    redX2   : reduced Chi squared error criteria - should be close to 1
    sigma_p : asymptotic standard error of the parameters
    sigma_y : asymptotic standard error of the curve-fit
    corr_p  : correlation matrix of the parameters
    R_sq    : R-squared cofficient of multiple determination  
    cvg_hst : convergence history (col 1: function calls, col 2: reduced chi-sq,
              col 3 through n: parameter values). Row number corresponds to
              iteration number.
    """

    global iteration, func_calls
    
    # iteration counter
    iteration  = 0
    # running count of function evaluations
    func_calls = 0

    # number of parameters
    Npar   = len(p)
    # number of data points
    Npnt   = len(y_dat)
    # previous set of parameters
    p_old  = np.zeros((Npar,1))
    # previous model, y_old = y_hat(t,p_old)
    y_old  = np.zeros((Npnt,1))
    # Jacobian matrix
    J      = np.zeros((Npnt,Npar))
    # statistical degrees of freedom
    DoF    = np.array([[Npnt - Npar + 1]])

    # weights or a scalar weight value ( weight >= 0 )
    weight = 1/(np.dot(y_dat.T,y_dat))
    # fractional increment of 'p' for numerical derivatives
    dp = [-0.001]      
    # lower bounds for parameter values
    p_min = -100*abs(p)  
    # upper bounds for parameter values       
    p_max = 100*abs(p)

    MaxIter       = 10000        # maximum number of iterations
    epsilon_1     = 1e-7        # convergence tolerance for gradient
    epsilon_2     = 1e-7        # convergence tolerance for parameters
    epsilon_4     = 1e-7        # determines acceptance of a L-M step
    lambda_0      = 1e-2        # initial value of damping paramter, lambda
    lambda_UP_fac = 11          # factor for increasing lambda
    lambda_DN_fac = 9           # factor for decreasing lambda
    Update_Type   = 1           # 1: Levenberg-Marquardt lambda update

    if len(dp) == 1:
        dp = dp*np.ones((Npar,1))

    idx   = np.arange(len(dp))  # indices of the parameters to be fit
    stop = 0                    # termination flag

    # initialize Jacobian with finite difference calculation
    JtWJ,JtWdy,X2,y_hat,J = lm_matx(t,p_old,y_old,1,J,p,y_dat,weight,dp,func)
    if np.abs(JtWdy).max() < epsilon_1:
        print('*** Your Initial Guess is Extremely Close to Optimal ***')
    
    lambda_0 = np.atleast_2d([lambda_0])

    # Marquardt: init'l lambda
    if Update_Type == 1:
        lambda_  = lambda_0
    
    # previous value of X2 
    X2_old = X2
    # initialize convergence history
    cvg_hst = np.ones((MaxIter,Npar+3))   
    
    # -------- Start Main Loop ----------- #
    while not stop and iteration <= MaxIter:
        
        iteration = iteration + 1
 
        # incremental change in parameters
        # Marquardt
        if Update_Type == 1:
            h = np.linalg.solve((JtWJ + lambda_*np.diag(np.diag(JtWJ)) ), JtWdy)  

        # update the [idx] elements
        p_try = p + h[idx]
        # apply constraints                             
        p_try = np.minimum(np.maximum(p_min,p_try),p_max)       
    
        # residual error using p_try
        delta_y = np.array([y_dat - lm_func(t,p_try,func)]).T
        
        # floating point error; break       
        if not all(np.isfinite(delta_y)):                   
          stop = 1
          break     

        func_calls = func_calls + 1
        # Chi-squared error criteria
        X2_try = np.dot(delta_y.T, ( delta_y * weight ))
  
        rho = np.matmul( np.dot(h.T, (lambda_ * h + JtWdy)),np.linalg.inv(X2 - X2_try))
    
        # it IS significantly better
        if ( rho > epsilon_4 ):                         
    
            dX2 = X2 - X2_old
            X2_old = X2
            p_old = p
            y_old = y_hat
            # % accept p_try
            p = p_try                        
        
            JtWJ,JtWdy,X2,y_hat,J = lm_matx(t,p_old,y_old,dX2,J,p,y_dat,weight,dp,func)
            
            # % decrease lambda ==> Gauss-Newton method
            # % Levenberg
            if Update_Type == 1:
                lambda_ = max(lambda_/lambda_DN_fac,1.e-7)

        # it IS NOT better
        else:                                           
            # % do not accept p_try
            X2 = X2_old
    
            if not np.remainder(iteration,2*Npar):            
                JtWJ,JtWdy,dX2,y_hat,J = lm_matx(t,p_old,y_old,-1,J,p,y_dat,weight,dp,func)
    
            # % increase lambda  ==> gradient descent method
            # % Levenberg
            if Update_Type == 1:
                lambda_ = min(lambda_*lambda_UP_fac,1.e7)

        # update convergence history ... save _reduced_ Chi-square
        cvg_hst[iteration-1,0] = func_calls
        cvg_hst[iteration-1,1] = X2/DoF
        cvg_hst[iteration-1,2] = lambda_
        
        for i in range(Npar):
            cvg_hst[iteration-1,i+3] = p.T[0][i]

        if ( max(abs(JtWdy)) < epsilon_1  and  iteration > 2 ):
          print('**** Convergence in r.h.s. ("JtWdy")  ****')
          stop = 1
    
        if ( max(abs(h)/(abs(p)+1e-12)) < epsilon_2  and  iteration > 2 ): 
          print('**** Convergence in Parameters ****')
          stop = 1
    
        if ( iteration == MaxIter ):
          print('!! Maximum Number of Iterations Reached Without Convergence !!')
          stop = 1

        # --- End of Main Loop --- #
        # --- convergence achieved, find covariance and confidence intervals

    #  ---- Error Analysis ----
    #  recompute equal weights for paramter error analysis
    if np.var(weight) == 0:   
        weight = DoF/(np.dot(delta_y.T,delta_y)) * np.ones((Npnt,1))
      
    # % reduced Chi-square                            
    redX2 = X2 / DoF

    JtWJ,JtWdy,X2,y_hat,J = lm_matx(t,p_old,y_old,-1,J,p,y_dat,weight,dp,func)

    # standard error of parameters 
    covar_p = np.linalg.inv(JtWJ)
    sigma_p = np.sqrt(np.diag(covar_p)) 
    error_p = sigma_p/p
    
    sigma_y = sigma

    # parameter correlation matrix
    corr_p = covar_p / [np.dot(sigma_p,sigma_p.T)]
        
    # coefficient of multiple determination
    R_sq = np.correlate(y_dat, y_hat)
    R_sq = 0        

    # convergence history
    cvg_hst = cvg_hst[:iteration,:]
    
    print('\nLM fitting results:')
    for i in range(Npar):
        print('----------------------------- ')
        print('parameter      = p%i' %(i+1))
        print('fitted value   = %0.4f' % p[i,0])
        print('standard error = %0.2f %%' % error_p[i,0])
    
    return p,redX2,sigma_p,sigma_y,corr_p,R_sq,cvg_hst