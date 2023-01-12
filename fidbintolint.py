import sympy as sym
import numpy as np
import pandas as pd
import scipy.stats as st

def fidbintolint(x1, x2, n1, n2, FUN, m1 = None, m2 = None, alpha = 0.05, P = 0.99, side = 1, K = 1000, B = 1000):
    '''
Fiducial-Based Tolerance Intervals for the Function of Two Binomial 
Proportions

Description
    Provides 1-sided or 2-sided tolerance intervals for the function of two 
    binomial proportions using fiducial quantities.

Usage
    fidbintol.int(x1, x2, n1, n2, m1 = NULL, m2 = NULL, FUN, 
                  alpha = 0.05, P = 0.99, side = 1, K = 1000, 
                  B = 1000)   

Parameters
----------
    x1 : int
        A value of observed "successes" from group 1.
        
    x2 : int
        A value of observed "successes" from group 2.
        
    n1 : int
        The total number of trials for group 1.
        
    n2 : int
        The total number of trials for group 2.
        
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
        See Details for the definition of the fiducial quantity for a binomial 
        proportion. The default is 1000.
        
    B : int, optional
        The number of iterations used for the Monte Carlo algorithm which 
        determines the tolerance limits. The number of iterations should be at 
        least as large as 1000. The default is 1000.

Details
    If X is observed from a Bin(n,p) distribution, then the fiducial quantity 
    for p is Beta(X+0.5,n-X+0.5).

Returns
-------
    fidbintol returns an f string with two sections. The first section 
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
    Clopper, C. J. and Pearson, E. S. (1934), The Use of Confidence or 
        Fiducial Limits Illustrated in the Case of the Binomial, Biometrika, 
        26, 404–413.
        
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
    95%/99% 1-sided and 2-sided tolerance intervals for the difference between 
    binomial proportions.
    
        p1 = 0.2
        
        p2 = 0.4
        
        n1 = 200
        
        n2 = 200
        
        x1 = st.binom.rvs(size=1,n=n1,p=p1)
        
        x2 = st.binom.rvs(size=1,n=n2,p=p2)
        
        x = sym.Symbol('x')
        
        y = sym.Symbol('y')
        
        fn = x-y
        
        fidbintolint(x1=x1, x2=x2, n1=n1, n2=n2, FUN=fn, m1=500, m2=500,side=1)
        
        fidbintolint(x1=x1, x2=x2, n1=n1, n2=n2, FUN=fn, m1=500, m2=500,side=2)
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
    est = np.round(FUN(x1/n1,x2/n2),8) # == R
    Qp1 = st.beta.rvs(size=K,a=x1+0.5,b=n1-x1+0.5) #== R
    Qp2 = st.beta.rvs(size=K,a=x2+0.5,b=n2-x2+0.5) # == R
    #####
    TEMP1 = pd.DataFrame([np.quantile(FUN(st.binom.rvs(size=B,n=m1,p=Qp1[i])/m1,st.binom.rvs(size=B,n=m2,p=Qp2[i])/m2),[P]) for i in range(1000)]) #should it be range(K) instead of range(1000)?-- R does range(1000)
    TEMP2 = pd.DataFrame([np.quantile(FUN(st.binom.rvs(size=B,n=m1,p=Qp1[i])/m1,st.binom.rvs(size=B,n=m2,p=Qp2[i])/m2),[1-P]) for i in range(1000)]) #should it be range(K) instead of range(1000)?-- R does range(1000)
    #####
    lower = np.quantile(TEMP2, alpha)
    upper = np.quantile(TEMP1, 1-alpha)
    pi_1 = sym.Symbol('pi_1')
    pi_2 = sym.Symbol('pi_2')
    F = F.subs({var[1]:pi_1, var[0]: pi_2}) #replace x and y with pi_1 and pi_2
    if side == 2:
        alpha = alpha * 2
        P = (2*P)-1
        return f'Tolerance Limits \n {pd.DataFrame({"alpha":[alpha], "P":[P], "fn.est":est, "2-sided.lower":lower, "2-sided.upper":upper})} \n\nFunction\n {F}'
    else:
        return f'Tolerance Limits \n {pd.DataFrame({"alpha":[alpha], "P":[P], "fn.est":est, "1-sided.lower":lower, "1-sided.upper":upper})} \n\nFunction\n {F}'
    
# p1 = 0.2
# p2 = 0.4
# n1 = 200
# n2 = 200
# x1 = st.binom.rvs(size=1,n=n1,p=p1)
# x2 = st.binom.rvs(size=1,n=n2,p=p2)
# x = sym.Symbol('x')
# y = sym.Symbol('y')
# fn = x-y
# print(fidbintolint(x1=x1, x2=x2, n1=n1, n2=n2, FUN=fn,m1=500,m2=500,side = 1))


        