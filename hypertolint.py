import numpy as np
from pandas import DataFrame as df
import scipy.stats as st

def length(x):
    if type(x) == int or type(x) == float:
        return 1
    else:
        return len(x)

def hypertolint(x, n, N, m = None, alpha = 0.05, P = 0.99, side = 1, method = 'EX'):
    '''
Hypergeometric Tolerance Intervals

Description
    Provides 1-sided or 2-sided tolerance intervals for hypergeometric random 
    variables. From a sampling without replacement perspective, these limits 
    use the proportion of units from group A (e.g., "black balls" in an urn) 
    in a sample to bound the number of potential units drawn from group A in 
    a future sample taken from the universe.
    
    hypertol.int(x, n, N, m = NULL, alpha = 0.05, P = 0.99, 
             side = 1, method = ["EX", "LS", "CC"])
        
Parameters
----------
    x: int
        The number of units from group A in the sample. Can be a vector, in 
        which case the sum of x is used.
    
    n: int
        The size of the random sample of units selected.
    
    N: int
        The population size.
    
    m: int, optional
        The quantity of units to be sampled from the universe for a future 
        study. If m = None, then the tolerance limits will be constructed 
        assuming n for this quantity. The default is None. 

    alpha: float, optional
        The level chosen such that 1-alpha is the confidence level.
    
    P: float, optional
        The proportion of units from group A in future samples of size m to be 
        covered by this tolerance interval.
    
    side: 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively). The default is 1. 
    
    method: string, optional
        The method for calculating the lower and upper confidence bounds, 
        which are used in the calculation of the tolerance bounds. The default 
        method is "EX", which is an exact-based method. "LS" is the 
        large-sample method. "CC" gives a continuity-corrected version of the 
        large-sample method. The default is 'EX'.
    
Returns
-------
  hypertolint returns a data frame with the following qualities:
        
    alpha:
        The specified significance level.

    P:
        The proportion of units from group A in future samples of size m.

    rate:	
        The sampling rate determined by n/N.

    p.hat:	
        The proportion of units in the sample from group A, calculated by x/n.

    1-sided.lower:
        The 1-sided lower tolerance bound. This is given only if side = 1.

    1-sided.upper:
        The 1-sided upper tolerance bound. This is given only if side = 1.
        
    2-sided.lower:	
        The 2-sided lower tolerance bound. This is given only if side = 2.

    2-sided.upper:	
        The 2-sided upper tolerance bound. This is given only if side = 2.

Note
----
    As this methodology is built using large-sample theory, if the sampling 
    rate is less than 0.05, then a warning is generated stating that the 
    results are not reliable. Also, compare the functionality of this 
    procedure with the accsamp procedure, which is to determine a minimal 
    acceptance limit for a particular sampling plan.

References
----------
    Brown, L. D., Cai, T. T., and DasGupta, A. (2001), Interval Estimation for 
        a Binomial Proportion, Statistical Science, 16, 101–133.
        
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Eichenberger, P., Hulliger, B., and Potterat, J. (2011), Two Measures for 
        Sample Size Determination, Survey Research Methods, 5, 27–37.

    Young, D. S. (2014), Tolerance Intervals for Hypergeometric and Negative 
        Hypergeometric Variables, Sankhya: The Indian Journal of Statistics, 
        Series B, 77(1), 114–140.
    
Examples
--------
    ## 90%/95% 1-sided and 2-sided hypergeometric tolerance intervals for a 
    future sample of 30 when the universe is of size 100.

    hypertol.int(x = 15, n = 50, N = 100, m = 30, alpha = 0.10, 
             P = 0.95, side = 1, method = "LS")
    hypertol.int(x = 15, n = 50, N = 100, m = 30, alpha = 0.10, 
             P = 0.95, side = 1, method = "CC")
    hypertol.int(x = 15, n = 50, N = 100, m = 30, alpha = 0.10, 
             P = 0.95, side = 2, method = "LS")
    hypertol.int(x = 15, n = 50, N = 100, m = 30, alpha = 0.10, 
             P = 0.95, side = 2, method = "CC")
    '''
    if (side != 1 and side != 2):
        return "Must specify a one-sided or two-sided procedure!"
    if side == 2:
        alpha = alpha/2
        P = (P+1)/2
    rate = n/N
    if rate < 0.05:
        print('Sampling rate is less than 0.05. The results',
              'may not be accurate!')
    if length(x) > 1:
        x = sum(x)
    phat = x/n
    k = st.norm.ppf(1-alpha)
    if m == None:
        m = n 
    fpc = np.sqrt((N-n)/(N-1))
    if method == 'EX':
        temp = np.array(range(0,N-x+1))
        lowerex =np.array(sorted(1-st.hypergeom.cdf(x-1,N, N-temp, n))) 
        tmp1 = np.where(lowerex > alpha)[0]
        tmp2 = np.array((length(temp)))
        tmp3 = min(tmp1)
        tmp4 = min(tmp2,tmp3)
        temp = temp+x
        Ml = temp[tmp4]
        upperex = np.array(sorted(st.hypergeom.cdf(x, N, temp, n),reverse=True))
        tmp1 = np.where(upperex > alpha)[0]
        tmp3 = max(tmp1)
        tmp4 = max(1,tmp3)
        Mu = temp[tmp4]
    elif method == 'LS' or method == 'CC':
        lowerp = phat - k*np.sqrt(phat*(1-phat)/n)*fpc - 1/(2 * n)*(method == "CC")
        upperp = phat + k*np.sqrt(phat*(1-phat)/n)*fpc + 1/(2 * n)*(method == "CC")
        lowerp = max(0,lowerp)
        upperp = min(upperp,1)
        Ml = max(0,np.floor(N*lowerp))
        Mu = min(np.ceil(N*upperp), N)
    lower = st.hypergeom.ppf(1-P, M = N, N = m, n = Ml) # == qhyper(p=1 - P, m = Ml,n = N - Ml,k = m)
    upper = st.hypergeom.ppf(P, M = N, N = m, n = Mu) # == qhyper(p=P, m = Mu, n = N - Mu, k = m)
    if side == 2:
        alpha = 2 * alpha
        P = (2 * P) - 1
        return df({"alpha":[alpha], "P":[P], "rate":[rate], "p.hat":[phat], "2-sided.lower":lower, "2-sided.upper":upper})
    else:
        return df({"alpha":[alpha], "P":[P], "rate":[rate], "p.hat":[phat], "1-sided.lower":lower, "1-sided.upper":upper})
        
print(hypertolint(x = [3,45], n = 532, N = 1435, m = None, alpha = 0.01, P = 0.99, side = 2, method = "LS"))






hypert















