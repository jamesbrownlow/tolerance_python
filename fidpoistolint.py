import sympy as sym
import numpy as np
import pandas as pd
import scipy.stats as st

def fidpoistolint(x1, x2, n1, n2, FUN, m1 = None, m2 = None, alpha = 0.05, P = 0.99, side = 1, K = 1000, B = 1000):
    '''
Fiducial-Based Tolerance Intervals for the Function of Two Poisson Rates

Description
    Provides 1-sided or 2-sided tolerance intervals for the function of two 
    Poisson rates using fiducial quantities.

Usage
    fidpoistolint(x1, x2, n1, n2, FUN, m1 = None, m2 = None, alpha = 0.05, 
                  P = 0.99, side = 1, K = 1000, B = 1000) 

Parameters
----------
    x1 : int
        A value of observed counts from group 1.
        
    x2 : int
        A value of observed counts from group 2.
        
    n1 : float
        The length of time that x1 was recorded over.
        
    n2 : float
        The length of time that x2 was recorded over.
        
    FUN : Sympy Symbolic Function
        Any reasonable (and meaningful) function taking exactly two arguments 
        that we are interested in constructing a tolerance interval on.
        
    m1 : int, optional
        The total number of future trials for group 1. If None, then it is set
        to n1. The default is None.
        
    m2 : int, optional
        The total number of future trials for group 2. If None, then it is set 
        to n2. The default is None.
        
    alpha : float, optional
        The level chosen such that 1-alpha is the confidence level. The 
        default is 0.05.
        
    P : float, optional
        The proportion of the population to be covered by this tolerance 
        interval. The default is 0.99.
        
    side : 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively). The default is 1.
        
    K : int, optional
        The number of fiducial quantities to be generated. The number of 
        iterations should be at least as large as the default value of 1000. 
        See Details for the definition of the fiducial quantity for a Poisson 
        rate.

    B : int, optional
        The number of iterations used for the Monte Carlo algorithm which 
        determines the tolerance limits. The number of iterations should be at 
        least as large as the default value of 1000

Details
    If X is observed from a Poi(n*λ) distribution, then the fiducial quantity 
    for λ is χ^{2}_{2*x+1}/(2*n).

Returns
-------
    fidpoistolint returns an f string with two sections. The first section 
    (Tolerance Limits) is a dataframe with the following items:
        alpha :
            The specified significance level.

        P :
            The proportion of the population covered by this tolerance interval.

        fn.est :
            A point estimate of the functional form of interest using the 
            maximum likelihood estimates calculated with the inputted values 
            of x1, x2, n1, and n2.

        1-sided.lower :
            The 1-sided lower tolerance bound. This is given only if side = 1.

        1-sided.upper :
            The 1-sided upper tolerance bound. This is given only if side = 1.

        2-sided.lower :
            The 2-sided lower tolerance bound. This is given only if side = 2.

        2-sided.upper :
            The 2-sided upper tolerance bound. This is given only if side = 2.
    
    The second section (Function) simply returns the functional form specified 
    by FUN. 

References
----------
    Cox, D. R. (1953), Some Simple Approximate Tests for Poisson Variates, 
        Biometrika, 40, 354–360.
        
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.

    Krishnamoorthy, K. and Lee, M. (2010), Inference for Functions of 
        Parameters in Discrete Distributions Based on Fiducial Approach: 
        Binomial and Poisson Cases, Journal of Statistical Planning and 
        Inference, 140, 1182–1192.

    Mathew, T. and Young, D. S. (2013), Fiducial-Based Tolerance Intervals for 
        Some Discrete Distributions, Computational Statistics and Data 
        Analysis, 61, 38–49.

Examples
--------
    ## 95%/99% 1-sided and 2-sided tolerance intervals for the ratio of two 
    Poisson rates.
    
    lambda1 = 10
    
    lambda2 = 2
    
    n1 = 3000
    
    n2 = 3250
    
    x1 = st.chi2.rvs(n1 * lambda1,size=1)
    
    x2 = st.chi2.rvs(n2 * lambda2,size=1)

    x = sym.Symbol('x')
    
    y = sym.Symbol('y')
    
    fn = x / y
    
    fidpoistolint(x1, x2, n1, n2, m1 = 2000, m2 = 2500, FUN = fn, alpha = 0.05,
                  P = 0.99, side = 1)
    
    fidpoistolint(x1, x2, n1, n2, m1 = 2000, m2 = 2500, FUN = fn, alpha = 0.05,
                  P = 0.99, side = 2)
    '''
    if side != 1 and side != 2:
        return 'Must specify a one-sided or two-sided procedure'
    if side == 2:
        alpha = alpha/2
        P = (P+1)/2
    if m1 == None:
        m1 = n1
    if m2 == None:
        m2 = n2
    var = list(FUN.atoms(sym.Symbol)) #gets the name of variables , x = var[1] and y = var[0]
    F = FUN #keeps symbolic notation used for output
    FUN = sym.lambdify([var[1],var[0]],FUN,'numpy') #makes it so it can have 2 inputs
    est = np.round(FUN(x1/n1,x2/n2),6)
    Ql1 = st.chi2.rvs(2*x1+1,size = K)/(2*n1) #== R
    Ql2 = st.chi2.rvs(2*x2+1,size = K)/(2*n2) #== R
    #####
    TEMP1 = pd.DataFrame([np.quantile(FUN(st.poisson.rvs(m1*Ql1[i],size = B)/m1,st.poisson.rvs(m2*Ql2[i],size = B)/m2),[P]) for i in range(1000)]) #should it be range(K) instead of range(1000)?-- R does range(1000)
    TEMP2 = pd.DataFrame([np.quantile(FUN(st.poisson.rvs(m1*Ql1[i],size = B)/m1,st.poisson.rvs(m2*Ql2[i],size = B)/m2),[1-P]) for i in range(1000)]) #should it be range(K) instead of range(1000)?-- R does range(1000)
    #####
    lower = np.quantile(TEMP2, alpha)
    upper = np.quantile(TEMP1, 1-alpha)
    lambda_1 = sym.Symbol('lambda_1')
    lambda_2 = sym.Symbol('lambda_2')
    F = F.subs({var[1]:lambda_1, var[0]: lambda_2}) #replace x and y with pi_1 and pi_2
    if side == 2:
        alpha = alpha * 2
        P = (2*P)-1
        return f'Tolerance Limits \n {pd.DataFrame({"alpha":[alpha], "P":[P], "fn.est":est, "2-sided.lower":lower, "2-sided.upper":upper})} \n\nFunction\n {F}\n'
    else:
        return f'Tolerance Limits \n {pd.DataFrame({"alpha":[alpha], "P":[P], "fn.est":est, "1-sided.lower":lower, "1-sided.upper":upper})} \n\nFunction\n {F}\n'

# lambda1 = 10
# lambda2 = 2
# n1 = 3000
# n2 = 3250
# x1 = st.chi2.rvs(n1 * lambda1,size=1)
# x2 = st.chi2.rvs(n2 * lambda2,size=1)
# x = sym.Symbol('x')
# y = sym.Symbol('y')
# fn = x / y

# print(fidpoistolint(x1, x2, n1, n2, m1 = 2000, m2 = 2500, FUN = fn, alpha = 0.05, P = 0.99, side = 1))
# print(fidpoistolint(x1, x2, n1, n2, m1 = 2000, m2 = 2500, FUN = fn, alpha = 0.05, P = 0.99, side = 2))