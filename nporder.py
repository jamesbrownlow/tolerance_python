import pandas as pd
import numpy as np
import scipy.stats as st
import scipy.optimize as opt

def nporder(m, alpha = 0.05, P = 0.99, indices = False):
    '''
    Sample Size Determination for Tolerance Limits Based on Order Statistics

Description
    For given values of m, alpha, and P, this function solves the necessary 
    sample size such that the r-th (or (n-s+1)-th) order statistic is the 
    [100(1-alpha)%, 100(P)%] lower (or upper) tolerance limit (see the Details
    section below for further explanation). This function can also report all 
    combinations of order statistics for 2-sided intervals.
    
    nporder(m, alpha = 0.05, P = 0.99, indices = FALSE)
        
Parameters
----------
    m: int
        See the Details section below for how m is defined.
    
    alpha: float, optional
        1 minus the confidence level attained when it is desired to cover a 
        proportion P of the population with the order statistics. The default 
        is 0.05.
    
    P: float, optional
        The proportion of the population to be covered with confidence 1-alpha 
        with the order statistics. The default is 0.99.
    
    indices: bool, optional
        An optional argument to report all combinations of order statistics 
        indices for the upper and lower limits of the 2-sided intervals. Note 
        that this can only be calculated when m>1. The default is False.
    
Details
    For the 1-sided tolerance limits, m=s+r such that the probability is at 
    least 1-alpha that at least the proportion P of the population is below 
    the (n-s+1)-th order statistic for the upper limit or above the r-th order 
    statistic for the lower limit. This means for the 1-sided upper limit that
    r=1, while for the 1-sided lower limit it means that s=1. For the 2-sided 
    tolerance intervals, m=s+r such that the probability is at least 1-alpha 
    that at least the proportion P of the population is between the r-th and 
    (n-s+1)-th order statistics. Thus, all combinations of r>0 and s>0 such 
    that m=s+r are considered.
    
Returns
-------
    If indices = FALSE, then a single number is returned for the necessary 
    sample size such that the r-th (or (n-s+1)-th) order statistic is the 
    [100(1-alpha)%, 100(P)%] lower (or upper) tolerance limit. If indices = 
    TRUE, then a list is returned with a single number for the necessary 
    sample size and a matrix with 2 columns where each row gives the pairs of 
    indices for the order statistics for all permissible 
    [100(1-alpha)%, 100(P)%] 2-sided tolerance intervals.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Hanson, D. L. and Owen, D. B. (1963), Distribution-Free Tolerance Limits 
        Elimination of the Requirement That Cumulative Distribution Functions 
        Be Continuous, Technometrics, 5, 518–522.

    Scheffe, H. and Tukey, J. W. (1945), Non-Parametric Estimation I. 
        Validation of Order Statistics, Annals of Mathematical Statistics, 16,
        187–192.
    
Examples
--------
    ## Only requesting the sample size.

        np.order(m = 5, alpha = 0.05, P = 0.95)

    ## Requesting the order statistics indices as well.

        np.order(m = 5, alpha = 0.05, P = 0.95, indices = TRUE)
    '''
    def f(n, m, alpha, P):
        return st.beta.cdf(1-P, m, n-m+1)-(1-alpha)
    n = np.ceil(opt.brentq(f=f,a=m,b=1e5*m,args=(m,alpha,P)))
    if indices == False or m < 2:
        return f'Sample size needed: {int(n)}'
    else:
        ind = list([0])
        ind = pd.DataFrame({'0': ind[0], '1': [n-(m-0)+2]})
        for i in range(1,m-1):
            ind.loc[len(ind.index)] = [i,n-(m-i)+2]
        return f'Sample size needed: {int(n)} \n\nOrder statistics\n{ind}'
