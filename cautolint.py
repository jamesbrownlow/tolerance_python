import scipy.optimize as opt
import pandas as pd
import numpy as np
import scipy.stats as st

def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    return len(x)

def cautolint(x, alpha = 0.05, P = 0.99, side = 1):
    '''
Cauchy Tolerance Intervals

Description
    Provides 1-sided or 2-sided tolerance intervals for Cauchy distributed 
    data.

Usage
    cautol.int(x, alpha = 0.05, P = 0.99, side = 1)

Parameters
----------
    x : 
        A vector of data which is Cauchy distributed.
        
    alpha : flaot, optional
        The level chosen such that 1-alpha is the confidence level. The 
        default is 0.05.
        
    P : float, optional
        The proportion of the population to be covered by this tolerance 
        interval. The default is 0.99.
        
    side : 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively). The default is 1.

Returns
-------
    cautol.int returns a data.frame with items:
        alpha :
            The specified significance level.

        P :
            The proportion of the population covered by this tolerance interval.

        1-sided.lower :
            The 1-sided lower tolerance bound. This is given only if side = 1.
            
        1-sided.upper :
            The 1-sided upper tolerance bound. This is given only if side = 1.

        2-sided.lower :
            The 2-sided lower tolerance bound. This is given only if side = 2.

        2-sided.upper :
            The 2-sided upper tolerance bound. This is given only if side = 2.


References
----------
    Bain, L. J. (1978), Statistical Analysis of Reliability and Life-Testing 
        Models, Marcel Dekker, Inc.
        
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.

Example
-------
    ## 95%/90% 2-sided Cauchy tolerance interval for a sample of size 1000. 

        x = np.random.standard_cauchy(size = 1000)
        
        out = cautolint(x = x, alpha = 0.05, P = 0.90, side = 2)
    '''
    if side != 1 and side != 2:
        return 'Must specify a one-sided or two-sided procedure'
    if side == 2:
        alpha = alpha/2
        P = (P+1)/2
    n = length(x)
    #x = np.array(x)
    inits = [np.median(x), st.iqr(x)/2]
    def caull(pars,x):
        return sum(-st.cauchy.logpdf(x, loc = pars[0], scale = pars[1]))
    out = (opt.minimize(caull,x0=inits, args = (x),method = 'BFGS')['x']) 
    thetahat = out[0]
    sigmahat = out[1]
    cfactor = 2 + 2*(st.cauchy.ppf(1-P))**2
    k = np.sqrt(cfactor/n)*st.norm.ppf(1-alpha) - st.cauchy.ppf(1-P)
    lower = thetahat-k*sigmahat
    upper = thetahat+k*sigmahat
    if side == 2:
        alpha = 2*alpha
        P = (2*P)-1
        return pd.DataFrame({"alpha":[alpha], "P":[P], "2-sided.lower":lower, "2-sided.upper":upper})
    else:
        return pd.DataFrame({"alpha":[alpha], "P":[P], "1-sided.lower":lower,"1-sided.upper":upper})
    

#x = [31,20,20,27,26,26,30,25,24,29,111,2,3,4,2,5,6,777,3,223,425,151,100,1000]
#x=np.random.standard_cauchy(size = 1000)
#print(cautolint(x = x, alpha = 0.05, P = 0.90, side = 1))

