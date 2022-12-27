import numpy as np
import scipy.stats
import pandas as pd

def uniftolint(x, alpha = 0.05, P = 0.99, upper = None, lower = None, side = 1):
    '''
Arguments
    x: A vector of data which is distributed according to a uniform distribution.
    
    alpha: The level chosen such that 1-alpha is the confidence level.
    
    P: The proportion of the population to be covered by this tolerance interval.
    
    upper: The upper bound of the data. When NULL, then the maximum of x is used.
    
    lower: The lower bound of the data. When NULL, then the minimum of x is used.
    
    side: Whether a 1-sided or 2-sided tolerance interval is required (determined by side = 1 or side = 2, respectively).
    
Value
  uniftol.int returns a data frame with items:
        
    alpha: The specified significance level.
    
    P: The proportion of the population covered by this tolerance interval.
    
    1-sided.lower: The 1-sided lower tolerance bound. This is given only if side = 1.
    
    1-sided.upper: The 1-sided upper tolerance bound. This is given only if side = 1.
    
    2-sided.lower: The 2-sided lower tolerance bound. This is given only if side = 2.
    
    2-sided.upper: The 2-sided upper tolerance bound. This is given only if side = 2.

References
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance Intervals. 
        Journal of Statistical Software, 36(5), 1-39. URL http://www.jstatsoft.org/v36/i05/.
    Faulkenberry, G. D. and Weeks, D. L. (1968), 
        Sample Size Determination for Tolerance Limits, Technometrics, 10, 343–348.
    '''
    if side != 1 and side != 2:
        return "Must specify a one-sided or two-sided procedure"
    if side == 2:
        alpha = alpha/2
        P = (P+1)/2
    n = len(x)
    x1 = lower
    xn = upper
    if x1 == None:
        x1 = min(x)
    if xn == None:
        xn = max(x)
    lower = ((1-P)/(1-alpha)**(1/n))*(xn-x1)+x1
    upper = (P/(alpha)**(1/n))*(xn-x1) + x1
    if side == 2:
        alpha *= 2
        P = (2*P)-1
        d = {"alpha":[alpha], "P":[P], "2-sided.lower":[lower], "2-sided.upper":[upper]}
        return pd.DataFrame(d)
    d = {"alpha":[alpha], "P":[P], "1-sided.lower":[lower], "1-sided.upper":[upper]}
    return pd.DataFrame(d)
