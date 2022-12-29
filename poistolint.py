import numpy as np
import pandas as pd
import scipy.stats

def poistolint(x, n, m = None, alpha = 0.05, P = 0.99, side = 1, method ='TAB'):
    '''
------------------------------------------------------------------------------
    
poistolint(x, n, m = None, alpha = 0.05, P = 0.99, side = 1, method = ["TAB", "LS", "SC", "CC", "VS", "RVS","FT", "CSC"])

Description
    Provides 1-sided or 2-sided tolerance intervals for Poisson random 
    variables. From a statistical quality control perspective, these limits 
    bound the number of occurrences (which follow a Poisson distribution) 
    in a specified future time period.

Parameters
    ----------
    x: int, float, or list
        The number of occurrences of the event in time period n. Can be a 
        vector of length n, in which case the sum of x is used.
    n : float
        The time period of the original measurements.
    m : float, optional
        The specified future length of time. If m = None, then the tolerance 
        limits will be constructed assuming n for the future length of time.
        The default is None.
    alpha : float, optional
        The level chosen such that 1-alpha is the confidence level. 
        The default is 0.05.
    P: float, optional
        The proportion of occurrences in future time lengths of size m to be 
        covered by this tolerance interval. The default is 0.99.
    side: 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively). The default is 1.
    method: string, optional
        The method for calculating the lower and upper confidence bounds, 
        which are used in the calculation of the tolerance bounds. The default 
        method is "TAB", which is the tabular method and is usually preferred 
        for a smaller number of occurrences. "LS" gives the large-sample 
        (Wald) method, which is usually preferred when the number of 
        occurrences is x>20. "SC" gives the score method, which again is 
        usually used when the number of occurrences is relatively large. "CC" 
        gives a continuity-corrected version of the large-sample method. "VS" 
        gives a variance-stabilized version of the large-sample method. "RVS" 
        is a recentered version of the variance-stabilization method. "FT" is 
        the Freeman-Tukey method. "CSC" is the continuity-corrected version of 
        the score method. More information on these methods can be found in 
        the "References". The default is 'TAB'.

Returns
-------
    poistolint returns a dataframe with items:
        
        alpha: 
            The specified significance level.
        P: 
            The proportion of occurrences in future time periods of length m.
        lambda.hat: 
            The mean occurrence rate per unit time, calculated by x/n.
        1-sided.lower: 
            The 1-sided lower tolerance bound. This is given 
            only if side = 1.
        1-sided.upper: 
            The 1-sided upper tolerance bound. This is given 
            only if side = 1.
        2-sided.lower: 
            The 2-sided lower tolerance bound. This is given 
            only if side = 2.
        2-sided.upper: 
            The 2-sided upper tolerance bound. This is given 
            only if side = 2.
        
References
----------
    Barker, L. (2002), A Comparison of Nine Confidence Intervals for a Poisson 
        Parameter When the Expected Number of Events Is ≤ 5, The American 
        Statistician, 56, 85–89.
        
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Freeman, M. F. and Tukey, J. W. (1950), Transformations Related to the 
        Angular and the Square Root, Annals of Mathematical Statistics, 21, 
        607–611.

    Hahn, G. J. and Chandra, R. (1981), Tolerance Intervals for Poisson and 
        Binomial Variables, Journal of Quality Technology, 13, 100–110.
        
Examples
--------
    ## 95%/90% 1-sided Poisson tolerance limits for future 
    ## occurrences in a period of length 3.

        poistol.int(x = 45, n = 9, m = 3, alpha = 0.05, P = 0.90, side = 1, 
            method = "TAB")
            
    ## 95%/90% 2-sided Poisson tolerance intervals for future 
    ## occurrences in a period of length 15 using 'LS' method.
    
        poistol.int(x = 45, n = 9, m = 15, alpha = 0.05, P = 0.90,
            side = 2, method = "LS")
    '''
    if (side != 1 and side != 2): 
        return "Must specify a one-sided or two-sided procedure"  
    if type(x) == int or type(x) == float:
        x = list([x])
    x = sum(x)
    if side ==2:
        alpha = alpha/2
        P = (P+1)/2
    if m == None:
        m = n
    if method == 'TAB':
        lowerlambda = 0.5* scipy.stats.chi2.ppf(alpha,2*x)/n
        upperlambda = 0.5* scipy.stats.chi2.ppf(1-alpha,2*x+2)/n
    elif method == 'LS':
        k = scipy.stats.norm.ppf(1-alpha)
        lowerlambda = (x/n) - (k*np.sqrt(x))/n
        upperlambda = (x/n) + (k*np.sqrt(x))/n
    elif method == 'SC':
        k = scipy.stats.norm.ppf(1-alpha)
        lowerlambda = (x/n) + (k**2/(2 * n)) - (k/np.sqrt(n)) * np.sqrt((x/n) + (k**2/(4 * n)))
        upperlambda = (x/n) + (k**2/(2 * n)) + (k/np.sqrt(n)) * np.sqrt((x/n) + (k**2/(4 * n)))
    elif method == 'CC':
        k = scipy.stats.norm.ppf(1-alpha)
        lowerlambda = (x/n) - (k * np.sqrt(x)/n + 0.5/n)
        upperlambda = (x/n) + (k * np.sqrt(x)/n + 0.5/n)
    elif method == 'VS':
        k = scipy.stats.norm.ppf(1-alpha)
        lowerlambda = (x/n) + (k**2/(4 * n)) - (k*np.sqrt(x)/n)
        upperlambda = (x/n) + (k**2/(4 * n)) + (k*np.sqrt(x)/n)
    elif method == 'RVS':
        k = scipy.stats.norm.ppf(1-alpha)
        lowerlambda = (x/n) + (k**2/(4 * n)) - (k*np.sqrt((x/n+3/8)/n))
        upperlambda = (x/n) + (k**2/(4 * n)) + (k*np.sqrt((x/n+3/8)/n))
    elif method == 'FT':
        def g(z):
            return ((z**2 - 1)/(2*z))**2
        k = scipy.stats.norm.ppf(1-alpha)
        TEMPL = np.sqrt(x/n) + np.sqrt((x/n) + 1) - k*(1/np.sqrt(n))
        TEMPU = np.sqrt(x/n) + np.sqrt((x/n) + 1) + k*(1/np.sqrt(n))
        if TEMPL >=1:
            lowerlambda = g(TEMPL)
        else:
            lowerlambda = 0
        upperlambda = g(TEMPU)
    elif method == 'CSC':
         k = scipy.stats.norm.ppf(1-alpha)
         lam = x/n
         lowerlambda = lam - (1/(2*n)) + k**2/(2*n) - np.sqrt((lam-1/(2*n)+k**2/(2*n))**2-lam**2+lam/n-1/(4*n**2))
         upperlambda = lam + (1/(2*n)) + k**2/(2*n) + np.sqrt((lam-1/(2*n)+k**2/(2*n))**2-lam**2+lam/n-1/(4*n**2))
    else:
        return 'Method entered not a method, retry'
    lowerlambda = max(0,lowerlambda)
    lower = scipy.stats.poisson.ppf(1-P, m*lowerlambda)
    upper = scipy.stats.poisson.ppf(P, m*upperlambda)
    if side == 2:
        alpha = 2*alpha
        P = (2*P)-1
        return pd.DataFrame({"alpha":[alpha], "P":[P], "lambda.hat":[x/n], "2-sided.lower":lower, "2-sided.upper":upper})
    else:
        return pd.DataFrame({"alpha":[alpha], "P":[P], "lambda.hat":[x/n], "1-sided.lower":lower, "1-sided.upper":upper})
    
help(poistolint)