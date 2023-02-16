import scipy.stats as st
import numpy as np
import pandas as pd
import scipy.optimize as opt
import warnings
warnings.filterwarnings(action = 'ignore')

'''
In the uppernu = min(TEMP2,0.9999999) codes, 1 was changed to 0.9999999
    due to a rounding error with 1. 
    
There are tons of comments in the code for if in the future we wanted x to 
    be allowed to be a list/array, but I am unsure if that even makes sense. 
    The R code allows for it, but it gives incorrect outputs. So, for now, x 
    must be a positive integer, 0, or a list of length n. 
'''

def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    return len(x)

def negbintolint(x, n, m = None, alpha = 0.05, P = 0.99, side = 1, method = 'LS'):
    '''
Negative Binomial Tolerance Intervals

Description
    Provides 1-sided or 2-sided tolerance intervals for negative binomial 
    random variables. From a statistical quality control perspective, these 
    limits use the number of failures that occur to reach n successes to bound 
    the number of failures for a specified amount of future successes (m).
    
    negbintolint(x, n, m = NULL, alpha = 0.05, P = 0.99, side = 1, method = 
                 "WU", "CB", "CS", "SC", "LR", "SP", "CC"])
    
Parameters
----------
    x : list
        The total number of failures that occur from a sample of size n. Can 
        be a vector of length n, in which case the sum of x is computed.
        
    n : int
        The target number of successes (sometimes called size) for each trial.
        
    m : int, optional
        The target number of successes in a future lot for which the tolerance 
        limits will be calculated. If m = NULL, then the tolerance limits will 
        be constructed assuming n for the target number of future successes. 
        The default is None.
        
    alpha : float, optional
        The level chosen such that 1-alpha is the confidence level. 
        The default is 0.05.
        
    P : float, optional
        The proportion of the defective (or acceptable) units in future 
        samples of size m to be covered by this tolerance interval. 
        The default is 0.99.
        
    side : 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively). The default is 1.
        
    method : string, optional
        The method for calculating the lower and upper confidence bounds, 
        which are used in the calculation of the tolerance bounds. 
            The default method is "LS", which is the large-sample method based 
            on the MLE. 
        
            "WU" is a Wald-type interval based on the UMVUE of the negative 
            binomial proportion. 
            
            "CB" is the Casella-Berger exact method. 
            
            "CS" is a method based on chi-square percentiles. 
            
            "SC" is the score method. 
            
            "LR" is a likelihood ratio-based method. 
            
            "SP" is a method using a saddlepoint approximation for the 
            confidence intervals. 
            
            "CC" gives a continuity-corrected version of the large-sample 
            method and is appropriate when n is large. More information on 
            these methods can be found in the "References".
            
Details
    This function takes the approach for Poisson and binomial random variables 
    developed in Hahn and Chandra (1981) and applies it to the negative 
    binomial case.

Returns
-------
    negbintolint returns a dataframe with items:
        
        alpha: The specified significance level.
        
        P: The proportion of defective (or acceptable) units in future 
        samples of size m.   
        
        pi.hat: The proportion of defective (or acceptable) units in the 
        sample, calculated by n/(n+x).
        
        1-sided.lower: The 1-sided lower tolerance bound. This is given 
        only if side = 1.
        
        1-sided.upper: The 1-sided upper tolerance bound. This is given 
        only if side = 1.
        
        2-sided.lower: The 2-sided lower tolerance bound. This is given 
        only if side = 2.
        
        2-sided.upper: The 2-sided upper tolerance bound. This is given 
        only if side = 2.
        
Note
----
    Recall that the geometric distribution is the negative binomial 
    distribution where the size is 1. Therefore, the case when n = m = 1 will 
    provide tolerance limits for a geometric distribution.
    
References
----------
    Casella, G. and Berger, R. L. (1990), Statistical Inference, Duxbury Press.
    
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Hahn, G. J. and Chandra, R. (1981), Tolerance Intervals for Poisson 
        and Binomial Variables, Journal of Quality Technology, 13, 100–110.

    Tian, M., Tang, M. L., Ng, H. K. T., and Chan, P. S. (2009), A Comparative 
        Study of Confidence Intervals for Negative Binomial Proportions, 
        Journal of Statistical Computation and Simulation, 79, 241–249.

    Young, D. S. (2014), A Procedure for Approximate Negative Binomial 
        Tolerance Intervals, Journal of Statistical Computation and Simulation, 
        84, 438–450.

Examples
--------
   ## Comparison of 95%/99% 1-sided tolerance limits with 50 failures before 
   10 successes are reached.
    
       negbintolint(x = 50, n = 10, side = 1, method = "LS")
       
       negbintolint(x = 50, n = 10, side = 1, method = "WU")
       
       negbintolint(x = 50, n = 10, side = 1, method = "CB")
       
       negbintolint(x = 50, n = 10, side = 1, method = "CS")
       
       negbintolint(x = 50, n = 10, side = 1, method = "SC")
       
       negbintolint(x = 50, n = 10, side = 1, method = "LR")
       
       negbintolint(x = 50, n = 10, side = 1, method = "SP")
       
       negbintolint(x = 50, n = 10, side = 1, method = "CC")
       
   ## 95%/99% 1-sided tolerance limits and 2-sided tolerance interval for the 
   same setting above, but when we are interested in a future experiment that 
   requires 20 successes be reached for each trial.
       
       negbintolint(x = 50, n = 10, m = 20, side = 1)
       
       negbintolint(x = 50, n = 10, m = 20, side = 2)

    '''
    if (side != 1) and (side != 2):
        return 'Must specify one sided or two sided procedure'
    if side == 2:
        alpha = alpha /2
        P = (P+1)/2
    if m == None:
        m = n
    if length(x) > 1 and length(x) != n:
        return 'x must be 0, a postive integer, or a vector of length n.'
    if length(x) == n:
        x = sum(x)
    nuhat = n/(n + x)
    nutilde = (n - 1)/(n + x - 1)
    za = st.norm.ppf(1 - alpha)
    senuhat = np.sqrt((nuhat**2 * (1 - nuhat))/n)
    if method == 'LS':
        maxtemp1 = np.max(nuhat-za*senuhat)
        mintemp2 = np.min(nuhat+za*senuhat)
        lowernu = max(1e-7, maxtemp1)
        uppernu = min(mintemp2, 0.9999999)
    elif method == 'WU':
        if x+n <= 2:
            return 'Bounds are not defined for this option. x+n <= 2'
        TEMP1 = nutilde - 1*za*np.sqrt(nutilde *(1-nutilde)/(n+x-2))
        TEMP2 = nutilde + 1*za*np.sqrt(nutilde *(1-nutilde)/(n+x-2))
        lowernu = max(1e-7, TEMP1)
        uppernu = min(TEMP2, 0.9999999)
    elif method == 'CB':   
        TEMP1 = n/(n + (x + 1) * st.f.ppf(1 - alpha, 2 * (x + 1), 2 * n))
        TEMP2 = n * st.f.ppf(1 - alpha, 2 * n, 2 * x)/(n * st.f.ppf(1 - alpha, 2 * n, 2 * x) + x)
        lowernu = max(1e-7, TEMP1)
        uppernu = min(TEMP2, 0.9999999) 
    elif method == 'CS':
        TEMP1 = st.chi2.ppf(alpha,2*n)/(2*(n+x))
        TEMP2 = st.chi2.ppf(1-alpha, 2*n)/(2*(n+x))
        lowernu = max(1e-07,TEMP1)
        uppernu = min(TEMP2,0.9999999) #if error in future, change 1 to 0.9999999
        
    elif method == 'SC':
        TEMP1 = ((2 * (n + x) * n - n * za**2) - 1 * np.sqrt(n**2 * za**4 - 4 * (n + x) * n**2 * za**2 + 4 * (n + x)**2 * n * za**2))/(2 * (n + x)**2)
        TEMP2 = np.min((2 * (n + x) * n - n * za**2) + 1 * np.sqrt(n**2 * za**4 - 4 * (n + x) * n**2 * za**2 + 4 * (n + x)**2 * n * za**2))/(2 * (n + x)**2)
        lowernu = max(1e-07,TEMP1)
        uppernu = min(TEMP2,0.9999999) #changed 1 to 0.9999999, round error was causing an issue
        
    elif method == 'LR':
        def funl(p, x, n, nuhat, alpha):
            return 2*(np.log(st.nbinom.pmf(x,n,nuhat))-np.log(st.nbinom.pmf(x,n,p)))-st.chi2.ppf(1-alpha,1)
        try:
            lowernu = opt.brentq(funl, a = 1e-12, b = nuhat, args=(x,n,nuhat,alpha), maxiter = 100000)
        except:
            lowernu = 1e-07
        def funu(p, x, n, nuhat, alpha):
            return 2 * (np.log(st.nbinom.pmf(x, n, nuhat)) - np.log(st.nbinom.pmf(x, n, p))) - st.chi2.ppf(alpha, 1)
        try:
            uppernu = opt.brentq(funu,a=nuhat,b=1,args=(x,n,nuhat,alpha),maxiter=100000)
        except:
            uppernu = 0.9999999
    
    elif method == 'SP':
        def funsp(p,x,n,Ka):
            theta = np.log(x/((n+x)*(1-p)))
            KY = n * (np.log(p) - np.log(1 - (1 - p) * np.exp(theta)))
            delta1 = np.sign(theta) * np.sqrt(2 * abs(theta * x - KY))
            delta2 = theta * np.sqrt(x * (1 + x/n))
            return st.norm.cdf(delta1) - st.norm.pdf(delta1) * ((1/delta2) - (1/delta1)) - Ka
        try:
            lowernu = opt.brentq(funsp, a = 1e-07, b = 0.9999999, args = (x,n,alpha),maxiter = 100000)
        except:
            lowernu = 1e-07
        lowernu = max(1e-07,lowernu)
        try:
            uppernu = opt.brentq(funsp, a = 1e-12, b = 0.9999999, args = (x,n,1-alpha),maxiter = 100000)
        except:
            uppernu = 1
        uppernu = min(1, uppernu)
        if x == 0:
            lowernu = 1
            uppernu = 1
    elif method == 'CC':
        lowernu = max(1e-07, nuhat - za * np.sqrt((nuhat**2 * (1 - nuhat))/n) - 0.5/(n + x))
        uppernu = min(nuhat + za * np.sqrt((nuhat**2 * (1 - nuhat))/n) + 0.5/(n + x), 1)
    lower = st.nbinom.ppf(1-P,n = m, p = uppernu)
    if lowernu <= 1e-07:
        upper = np.inf
    else:
        upper = st.nbinom.ppf(P,n = m, p = lowernu)
    if side == 2:
        alpha *= 2
        P = (2*P)-1
        temp = pd.DataFrame({"alpha":[alpha], "P":[P], "pi.hat":[np.round(nuhat,7)], "2-sided.lower":[lower], "2-sided.upper":[upper]})
        return temp
    if side == 1:
        temp = pd.DataFrame({"alpha":[alpha], "P":[P], "pi.hat":[np.round(nuhat,7)], "1-sided.lower":[lower], "1-sided.upper":[upper]})
        return temp
        
#tests
# x = [negbintolint(x = 50, n = 10, m=20, side = 1, method = "LS"),
# negbintolint(x = 50, n = 10, m=20, side = 1, method = "LS"),
# negbintolint(x = 50, n = 10, m=20, side = 1, method = "WU"),
# negbintolint(x = 50, n = 10, m=20, side = 1, method = "WU"),
# negbintolint(x = 50, n = 10, m=20, side = 1, method = "CB"),
# negbintolint(x = 50, n = 10, m=20, side = 1, method = "CB"),
# negbintolint(x = 50, n = 10, m=20, side = 1, method = "CS"),
# negbintolint(x = 50, n = 10, m=20, side = 1, method = "CS"),
# negbintolint(x = 50, n = 10, m=20, side = 1, method = "SC"),
# negbintolint(x = 50, n = 10, m=20, side = 1, method = "SC"),
# negbintolint(x = 50, n = 10, m=20, side = 1, method = "LR"),
# negbintolint(x = 50, n = 10, m=20, side = 1, method = "LR"),
# negbintolint(x = 50, n = 10, m=20, side = 1, method = "SP"),
# negbintolint(x = 50, n = 10, m=20, side = 1, method = "SP"),
# negbintolint(x = 50, n = 10, m=20, side = 1, method = "CC"),
# negbintolint(x = 50, n = 10, m=20, side = 1, method = "CC"),
# negbintolint(x = 50, n = 10, m=20, side = 2, method = "LS"),
# negbintolint(x = 50, n = 10, m=20, side = 2, method = "LS"),
# negbintolint(x = 50, n = 10, m=20, side = 2, method = "WU"),
# negbintolint(x = 50, n = 10, m=20, side = 2, method = "WU"),
# negbintolint(x = 50, n = 10, m=20, side = 2, method = "CB"),
# negbintolint(x = 50, n = 10, m=20, side = 2, method = "CB"),
# negbintolint(x = 50, n = 10, m=20, side = 2, method = "CS"),
# negbintolint(x = 50, n = 10, m=20, side = 2, method = "CS"),
# negbintolint(x = 50, n = 10, m=20, side = 2, method = "SC"),
# negbintolint(x = 50, n = 10, m=20, side = 2, method = "SC"),
# negbintolint(x = 50, n = 10, m=20, side = 2, method = "LR"),
# negbintolint(x = 50, n = 10, m=20, side = 2, method = "LR"),
# negbintolint(x = 50, n = 10, m=20, side = 2, method = "SP"),
# negbintolint(x = 50, n = 10, m=20, side = 2, method = "SP"),
# negbintolint(x = 50, n = 10, m=20, side = 2, method = "CC"),
# negbintolint(x = 50, n = 10, m=20, side = 2, method = "CC")]
# x = x[::2]
# for  a in x:
#     print(a)

