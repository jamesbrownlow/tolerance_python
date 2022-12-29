import numpy as np
import pandas as pd
import scipy.stats

def laptolint(x,alpha=0.05,P=0.99,side=1):
    ''' 
Laplace Tolerance Intervals   

Description
    Provides 1-sided or 2-sided tolerance intervals for data distributed 
    according to a Laplace distribution.

Parameters
----------
    x : list
        A vector of data which is distributed according to a Laplace 
        distribution.
    alpha : float, optional
        The level chosen such that 1-alpha is the confidence level. 
        The default is 0.05.
    P : float, optional
        The proportion of the population to be covered by this tolerance 
        interval. The default is 0.99.
    side : 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively). The default is 1.

Returns
-------
    Returns a dataframe with items:
        alpha:
            The specified significance level.
        P:	
            The proportion of the population covered by this tolerance interval.

        1-sided.lower	
            The 1-sided lower tolerance bound. This is given only if side = 1.
        
        1-sided.upper	
            The 1-sided upper tolerance bound. This is given only if side = 1.
        
        2-sided.lower	
            The 2-sided lower tolerance bound. This is given only if side = 2.
        
        2-sided.upper	
            The 2-sided upper tolerance bound. This is given only if side = 2.
            
References
----------
    Bain, L. J. and Engelhardt, M. (1973), Interval Estimation for the Two 
        Parameter Double Exponential Distribution, Technometrics, 15, 875–887.
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.

Examples
--------
    ##First generate data from a laplace distribution with loc = 70 scale = 3 and size = 40
        x = numpy.random.laplace(loc=70,scale=3,size=40)
    
    ## 95%/90% 1-sided Laplace tolerance intervals for the sample of size 40 generated above
        laptolint(x, alpha = 0.05, P = .90, side = 1)

    '''
    if side != 1 and side != 2:
        return 'Must specify a one-sided or two-sided procedure.'
    if len(list(x)) == 0 or len(list(x)) == 1:
        return 'Must have more than one element in your data.'
    if side == 2:
        alpha = alpha/2
        P = (P+1)/2
    n = len(x)
    kb = np.log(2*(1-P))
    muhat = np.median(x)
    betahat = np.mean(abs(x-np.median(x)))
    k = (-n*kb+scipy.stats.norm.ppf(1-alpha)*np.sqrt(n*(1+kb**2)-scipy.stats.norm.ppf(1-alpha)**2))/(n-scipy.stats.norm.ppf(1-alpha)**2)
    lower = muhat - k*betahat
    upper = muhat + k*betahat
    if side == 2:
        alpha = 2*alpha
        P = (2*P)-1
        return pd.DataFrame({"alpha":[alpha], "P":[P], "2-sided.lower": lower, "2-sided.upper": upper})
    else:
        return pd.DataFrame({"alpha":[alpha], "P":[P], "1-sided.lower": lower, "1-sided.upper": upper})

lapto

