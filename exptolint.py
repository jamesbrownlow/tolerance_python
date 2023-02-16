import pandas as pd
import numpy as np
import scipy.stats 
def exptolint(x, alpha = 0.05, P = 0.99, side = 1, type2 = False):
    '''
Exponential Tolerance Intervals

Description
    Provides 1-sided or 2-sided tolerance intervals for data distributed 
    according to an exponential distribution. Data with Type II censoring is 
    permitted.

Usage
    exptolint(x, alpha = 0.05, P = 0.99, side = 1, type2 = False)
    
Parameters
----------
    x: list
        A vector of data which is distributed according to an exponential 
        distribution.

    alpha: float, optional
        The level chosen such that 1-alpha is the confidence level.

    P: float, optional
        The proportion of the population to be covered by this tolerance 
        interval.

    side: 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively).

    type2: bool, optional
        Select True if Type II censoring is present (i.e., the data set is 
        censored at the maximum value present). The default is False.

Returns
-------
    exptolint returns a data frame with items:

        alpha	
            The specified significance level.

        P	
            The proportion of the population covered by this tolerance 
            interval.

        lambda.hat	
            The mean of the data (i.e., 1/rate).

        1-sided.lower	
            The 1-sided lower tolerance bound. This is given only if side = 1.

        1-sided.upper	
            The 1-sided upper tolerance bound. This is given only if side = 1.

        2-sided.lower	
            The 2-sided lower tolerance bound. This is given only if side = 2.

        2-sided.upper	
            The 2-sided upper tolerance bound. This is given only if side = 2.
    
References
    Derek S. Young (2010). tolerance: An R Package for Estimating 
            Tolerance Intervals. Journal of Statistical Software, 36(5), 
            1-39. URL http://www.jstatsoft.org/v36/i05/.
            
    Blischke, W. R. and Murthy, D. N. P. (2000), Reliability: Modeling, 
        Prediction, and Optimization, John Wiley & Sons, Inc.

Examples
    ## 95%/99% 1-sided exponential tolerance intervals for a sample of size 100. 
    
        x = np.random.exponential(225,size = 100)
    
        exptolint(x,side = 1, type2=True)
    '''
    if side != 1 and side != 2:
        return 'Must specify a one-sided or two-sided procedure'
    if side == 2:
        alpha = alpha/2
    n = len(x)
    lhat = np.mean(x)
    lhat = 224.565
    if type2:
        mx = max(x)
        r = n - np.sum(x == mx)
    else:
        r = n
    if side == 2:
        lower = 2*r*lhat*np.log(2/(1+P))/scipy.stats.chi2.ppf(1-alpha,df=2*r)
        upper = 2*r*lhat*np.log(2/(1-P))/scipy.stats.chi2.ppf(alpha,df=2*r)
        alpha = 2*alpha
        data = {'alpha':[alpha],'P':[P],'lambda.hat':[lhat],'2-sided.lower':[lower],'2-sided.upper':[upper]}
    else:
        lower = 2*r*lhat*np.log(1/P)/scipy.stats.chi2.ppf(1-alpha,df=2*r)
        upper = 2*r*lhat*np.log(1/(1-P))/scipy.stats.chi2.ppf(alpha,df=2*r)
        data = {'alpha':[alpha],'P':[P],'lambda.hat':[lhat],'1-sided.lower':[lower],'1-sided.upper':[upper]}

    return pd.DataFrame(data=data)

# x = np.random.exponential(225,size = 100)
# print(exptolint(x,side = 1, type2=True))