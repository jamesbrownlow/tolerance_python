import numpy as np
import pandas as pd
import scipy.stats
from scipy.special import digamma

def exttolint(x, alpha = 0.05, P = 0.99, side = 1, dist = 'Weibull', ext = 'min', NRdelta = 1e-8):
    '''
------------------------------------------------------------------------------
    
exttolint(x, alpha = 0.05, P = 0.99, side = 1, dist = ["Weibull","Gumbel"], ext = 'min', NRdelta = 1e-8)

Description
    Provides 1-sided or 2-sided tolerance intervals for data distributed 
    according to either a Weibull distribution or extreme-value 
    (also called Gumbel) distributions.

Details
Recall that the relationship between the Weibull distribution and the 
extreme-value distribution for the minimum is that if the random variable X is 
distributed according to a Weibull distribution, then the random variable 
Y = ln(X) is distributed according to an extreme-value distribution for the 
minimum. 

If dist = "Weibull", then the natural logarithm of the data are taken 
so that a Newton-Raphson algorithm can be employed to find the MLEs of the 
extreme-value distribution for the minimum and then the data and MLEs are 
transformed back appropriately. No transformation is performed if 
dist = "Gumbel". The Newton-Raphson algorithm is initialized by the method of 
moments estimators for the parameters.

Parameters
    ----------
    x: list
        A vector of data which is distributed according to either a Weibull 
        distribution or an extreme-value distribution.
    alpha : float, optional
        The level chosen such that 1-alpha is the confidence level. 
        The default is 0.05.
    P: float, optional
        The proportion of the population to be covered by this tolerance 
        interval. The default is 0.99.
    side: 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively). The default is 1.
    dist: string, optional
        Select either dist = "Weibull" or dist = "Gumbel" if the data is 
        distributed according to the Weibull or extreme-value distribution, 
        respectively. The default is 'Weibull'.
    ext: string, optional
        If dist = "Gumbel", then select which extreme is to be modeled for the 
        Gumbel distribution. The Gumbel distribution for the minimum 
        (i.e., ext = "min") corresponds to a left-skewed distribution and the 
        Gumbel distribution for the maximum (i.e., ext = "max") corresponds to
        a right-skewed distribution The default is 'min'.
    NRdelta: float, optional
        The stopping criterion used for the Newton-Raphson algorithm when 
        finding the maximum likelihood estimates of the Weibull or 
        extreme-value distribution. The default is 1e-8.

Returns
-------
    extolint returns a dataframe with items:
        alpha: 
            The specified significance level.
        
        P: 
            The proportion of the population covered by this tolerance interval.  
        
        shape1: 
            MLE for the shape parameter if dist = "Weibull" or for the 
            location parameter if dist = "Gumbel".
        
        shape2:
            MLE for the scale parameter if dist = "Weibull" or 
            dist = "Gumbel".
        
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
    Bain, L. J. and Engelhardt, M. (1981), Simple Approximate Distributional 
        Results for Confidence and Tolerance Limits for the Weibull 
        Distribution Based on Maximum Likelihood Estimators, Technometrics, 
        23, 15–20.
        
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.

    Coles, S. (2001), An Introduction to Statistical Modeling of Extreme 
        Values, Springer.

Examples
--------
    ## 85%/90% 1-sided Weibull tolerance intervals for a sample
    ## of size 150. 
        x = numpy.random.weibull(size = 150, loc = 3, scale = 75)
        exttolint(x, alpha = 0.15, P = 0.90, side = 1, dist = 'Weibull')
    '''
    if (side != 1 and side != 2): 
        return "Must specify a one-sided or two-sided procedure"  
    if type(x) == int or type(x) == float:
        return 'Must have more than one element in your data.'
    mx = abs(max(x))+1000
    tempind = 0
    count = 0
    if len([count+1 for y in x if abs(y)>1000]) > 0:
        tempind = 1
        x = [y/mx for y in x]
    if side == 2:
        alpha = alpha/2
        P = (P+1)/2
    n = len(x)
    if dist == 'Weibull':
        x = np.log(x)
        ext = 'min'
    delta = np.sqrt((np.mean([y**2 for y in x])-np.mean(x)**2)*6/np.pi**2)
    xbar = np.mean(x)
    temp = ((dist == 'Weibull') or (dist == 'Gumbel' and ext == 'min'))
    xi = xbar + digamma(1)*(1-2*temp)
    thetaold = [xi,delta]
    diff = [1]
    if temp == True:
        count = 0
        while ((diff[0]>NRdelta) > 0) or ((diff[1]>NRdelta) > 0):
            f = sum(x*np.exp(x/delta))
            f1 = -sum([y**2 * np.exp(y/delta) for y in x])/(delta**2)
            g = sum(np.exp(x/delta))
            g1 = -f/(delta**2)
            d = delta + xbar - (f/g)
            d1 = 1-(g*f1-f*g1)/(g**2)
            deltanew = delta - d/d1
            xinew = -deltanew *np.log(n/sum(np.exp(x/deltanew)))
            deltaold = delta
            xiold = xi
            delta = deltanew
            xi = xinew
            if xi == None or delta == None or delta < 0:
                xi = thetaold[0]
                delta = thetaold[1]
                diff = NRdelta/5
            else:
                diff = [abs(deltanew - deltaold), abs(xinew - xiold)]
    else:
        lam = 1/delta
        while ((diff[0]>NRdelta) > 0) or ((diff[1]>NRdelta) > 0):
            f = sum([y * np.exp(-lam * y) for y in x])
            f1 = -sum([y**2 * np.exp(-lam * y) for y in x])
            g = sum([np.exp(-lam * y) for y in x])
            g1 = -f
            d = (1/lam) - xbar + (f/g)
            d1 = (f**2/g**2) + (f1/g) - (1/lam**2)
            lamnew = lam - (d/d1)
            xinew = -(1/lamnew) * np.log((1/n) * sum([np.exp(-lamnew * y) for y in x]))
            lamold = lam
            xiold = xi
            deltaold = 1/lam
            lam = lamnew
            xi = xinew
            deltanew = 1/lam
            delta = deltanew
            if xi == None or delta == None or delta < 0:
                xi = thetaold[0]
                delta = thetaold[1]
                lam=1/delta
                diff = NRdelta/5
            else:
                diff = [abs(deltanew - deltaold), abs(xinew - xiold)]
    def lamb(P):
        return np.log(-np.log(P))
    def kt(x1, x2, n):
        return scipy.stats.nct.ppf(1-x1,df=n-1,nc=(-np.sqrt(n)*lamb(x2)))
    lower = xi - delta * kt(alpha, P, n)/np.sqrt(n - 1)
    upper = xi - delta * kt(1 - alpha, 1 - P, n)/np.sqrt(n - 1)
    if (dist == "Gumbel" and ext == "max"):
        #        lower <- xi + delta * k.t(alpha, 1 - P, n)/sqrt(n - 1)
        #        upper <- xi + delta * k.t(1 - alpha, P, n)/sqrt(n - 1)
        lower = xi + delta * kt(1 - alpha, 1 - P, n)/np.sqrt(n - 1)
        upper = xi + delta * kt(alpha, P, n)/np.sqrt(n - 1)
    a = xi
    b = delta
    if dist == "Weibull":
        a = 1/delta
        b = np.exp(xi)
        lower = np.exp(lower)
        upper = np.exp(upper)
    if side == 2:
        alpha = 2*alpha
        P = (2*P)-1
    if tempind == 1:
        b = b*mx
        lower = lower*mx
        upper = upper*mx
    if side == 2:
        return pd.DataFrame({"alpha":[alpha], "P":[P], "shape.1":[a], "shape.2":[b], "2-sided.lower":lower, "2-sided.upper":upper})
    else:
        return pd.DataFrame({"alpha":[alpha], "P":[P], "shape.1":[a], "shape.2":[b], "1-sided.lower":lower, "1-sided.upper":upper})


help(exttolint)
