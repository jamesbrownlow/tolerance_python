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
    try:
        n = len(x)
    except:
        n = 1
    x1 = lower
    xn = upper
    if x1 is None:
        try:
            x1 = min(x)
        except:
            x1 = x
    if xn is None:
        try:
            xn = max(x)
        except:
            xn=x
    lower = ((1 - P)/(1 - alpha)**(1/n)) * (xn - x1) + x1
    upper = (P/(alpha)**(1/n)) * (xn - x1) + x1
    if side == 2:
        alpha *= 2
        P = (2*P)-1
        d = {"alpha":[alpha], "P":[P], "2-sided.lower":[lower], "2-sided.upper":[upper]}
        return pd.DataFrame(d)
    d = {"alpha":[alpha], "P":[P], "1-sided.lower":[lower], "1-sided.upper":[upper]}
    return pd.DataFrame(d)

# x = [6, 2, 1, 4, 8, 3, 3, 14, 2, 1, 21, 5, 18, 2, 30, 10, 8, 2, 
#                   11, 4, 16, 13, 17, 1, 7, 1, 1, 28, 19, 27, 2, 7, 7, 13, 1,
#                   15, 1, 16, 9, 9, 7, 29, 3, 10, 3, 1, 20, 8, 12, 6, 11, 5, 1,
#                   5, 23, 3, 3, 14, 6, 9, 1, 24, 5, 11, 15, 1, 5, 5, 4, 10, 1,
#                   12, 1, 3, 4, 2, 9, 2, 1, 25, 6, 8, 2, 1, 1, 1, 4, 6, 7, 26, 
#                   10, 2, 1, 2, 17, 4, 3, 22, 8, 2]
# print(uniftolint(x=x,side = 1))