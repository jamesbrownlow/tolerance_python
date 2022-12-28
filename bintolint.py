import numpy as np
import scipy
import pandas
import scipy.stats

def bintolint(x, n, alpha = 0.05, P = 0.99, m = None, side = 1, method = 'LS', a1=0.5, a2=0.5):
    '''
    bintolint(x, n, m = NULL, alpha = 0.05, P = 0.99, side = 1, method = c("LS", "WS", "AC", "JF", "CP", "AS", "LO", "PR", "PO", "CL", "CC", "CWS"), a1 = 0.5, a2 = 0.5)
Parameters
    ----------
    x : list
        The number of defective (or acceptable) units in the sample. 
        Can be a vector of length n, in which case the sum of x is used.
    n : int
        The size of the random sample of units selected for inspection.
    alpha : float, optional
        The level chosen such that 1-alpha is the confidence level. 
        The default is 0.05.
    P : float, optional
        The proportion of the defective (or acceptable) units in future 
        samples of size m to be covered by this tolerance interval. 
        The default is 0.99.
    m : int, optional
        The quantity produced in future groups. If m = NULL, then the 
        tolerance limits will be constructed assuming n for this quantity. 
        The default is None.
    side : 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively). The default is 1.
    method : string, optional
        The method for calculating the lower and upper confidence bounds, 
        which are used in the calculation of the tolerance bounds. The default 
        method is "LS", which is the large-sample method. "WS" is Wilson's 
        method, which is just the score confidence interval. "AC" gives the 
        Agresti-Coull method, which is also appropriate when the sample size 
        is large. "JF" is Jeffreys' method, which is a Bayesian approach to 
        the estimation. "CP" is the Clopper-Pearson (exact) method, which is 
        based on beta percentiles and provides a more conservative interval. 
        "AS" is the arcsine method, which is appropriate when the sample 
        proportion is not too close to 0 or 1. "LO" is the logit method, 
        which also is appropriate when the sample proportion is not too 
        close to 0 or 1, but yields a more conservative interval. "PR" uses 
        a probit transformation and is accurate for large sample sizes. 
        "PO" is based on a Poisson parameterization, but it tends to be more 
        erratic compared to the other methods. "CL" is the complementary log 
        transformation and also tends to perform well for large sample sizes. 
        "CC" gives a continuity-corrected version of the large-sample method. 
        "CWS" gives a continuity-corrected version of Wilson's method. More 
        information on these methods can be found in the "References". 
        The default is 'LS'.
    a1 : int, optional
        This specifies the first shape hyperparameter when using Jeffreys' 
        method. The default is 0.5.
    a2 : int, optional
        This specifies the second shape hyperparameter when using Jeffreys' 
        method. The default is 0.5.

Returns
-------
    bintolint returns a dataframe with items:
        alpha: The specified significance level.
        
        P: The proportion of defective (or acceptable) units in future 
        samples of size m.   
        phat: The proportion of defective (or acceptable) units in the sample, 
        calculated by x/n.
        
        1-sided.lower: The 1-sided lower tolerance bound. This is given 
        only if side = 1.
        
        1-sided.upper: The 1-sided upper tolerance bound. This is given 
        only if side = 1.
        
        2-sided.lower: The 2-sided lower tolerance bound. This is given 
        only if side = 2.
        
        2-sided.upper: The 2-sided upper tolerance bound. This is given 
        only if side = 2.
        
References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance Intervals. 
        Journal of Statistical Software, 36(5), 1-39. URL http://www.jstatsoft.org/v36/i05/.
        
    Brown, L. D., Cai, T. T., and DasGupta, A. (2001), Interval Estimation 
        for a Binomial Proportion, Statistical Science, 16, 101–133.

    Hahn, G. J. and Chandra, R. (1981), Tolerance Intervals for Poisson and 
        Binomial Variables, Journal of Quality Technology, 13, 100–110.

    Newcombe, R. G. (1998), Two-Sided Confidence Intervals for the Single 
        Proportion: Comparison of Seven Methods, Statistics in Medicine, 17, 
        857–872.

Examples
--------
    ## Using Jeffreys' method to construct the 85%/90% 1-sided binomial tolerance limits.
        bintolint(x = 230, n = 1000, m = 2500, alpha = 0.15, P = 0.90, side = 1, method = "JF", a1 = 2, a2 = 10)
        
        bintolint(x = [1,0,1,1,0,1,1,0], n = 1000, m = 2500, alpha = 0.15, P = 0.90, side = 1, method = "JF", a1 = 2, a2 = 10)

    '''
    if(side != 1 and side != 2):
        return "must be one or two sided only"
    if(side == 2):
        alpha = alpha/2
        P = (P+1)/2
    if type(x) == int or type(x) == float:
        x = x
    else:
        x = sum(x)
    phat = x/n
    k = scipy.stats.norm.ppf(1-alpha)
    xtilde = (x+(k**2)/2)
    ntilde = (n+k**2)
    ptilde = xtilde/ntilde
    if m == None:
        m = n
    if method == 'LS':  
        lowerp = phat - k * np.sqrt(phat *(1-phat)/n)
        upperp = phat + k * np.sqrt(phat *(1-phat)/n)
    elif method == 'WS':
        lowerp = ptilde - (k*np.sqrt(n)/(n+k**2))*np.sqrt(phat*(1-phat)+(k**2/(4*n)))
        upperp = ptilde - (k*np.sqrt(n)/(n+k**2))*np.sqrt(phat*(1-phat)+(k**2/(4*n)))
    elif method == 'AC':
        lowerp = ptilde-k*np.sqrt(ptilde*(1-ptilde)/ntilde)
        upperp = ptilde+k*np.sqrt(ptilde*(1-ptilde)/ntilde)
    elif method == 'JF':
        lowerp = scipy.stats.beta.ppf(alpha, x + a1, n - x + a2)
        upperp = scipy.stats.beta.ppf(1-alpha, a = x+a1, b = n-x+a2)
    elif method == 'CP':
        lowerp = (1+((n-x+1)*scipy.stats.f.ppf(1-alpha,2*(n-x+1),(2*x)))/x)**(-1)
        upperp = (1+(n-x)/((x+1)*scipy.stats.f.ppf(1-alpha, 2*(x+1), 2*(n-x))))**(-1)
    elif method == 'AS':
        psin = (x+(3/8))/(n+(3/4))
        lowerp = (np.sin(np.arcsin(np.sqrt(psin))-0.5*k/np.sqrt(n)))**2
        upperp = (np.sin(np.arcsin(np.sqrt(psin))+0.5*k/np.sqrt(n)))**2
    elif method == 'LO':
        lhat = np.log(x/(n-x))
        Vhat = n/(x*(n-x))
        lowerlambda = lhat-k*np.sqrt(Vhat)
        upperlambda = lhat+k*np.sqrt(Vhat)
        lowerp = np.exp(lowerlambda)/(1+np.exp(lowerlambda))
        upperp = np.exp(upperlambda)/(1+np.exp(upperlambda))
    elif method == 'PR':
        zhat = scipy.stats.norm.ppf(phat)
        lowerp = scipy.stats.norm.cdf(zhat-1*k*np.sqrt((phat*(1-phat))/(n*scipy.stats.norm.pdf(zhat)**2)))
        upperp = scipy.stats.norm.cdf(zhat+1*k*np.sqrt((phat*(1-phat))/(n*scipy.stats.norm.pdf(zhat)**2)))
    elif method == 'PO':
        muhat = -np.log(phat)
        lowerp = np.exp(-(muhat+1*k*np.sqrt((1-phat)/n*phat)))
        upperp = np.exp(-(muhat-1*k*np.sqrt((1-phat)/n*phat)))
    elif method == 'CL':
        muhat = -np.log(phat)
        gammahat = np.log(muhat)
        lowerp = np.exp(-np.exp(gammahat+1*k*np.sqrt((1-phat)/(n*phat*muhat**2))))
        upperp = np.exp(-np.exp(gammahat-1*k*np.sqrt((1-phat)/(n*phat*muhat**2))))
    elif method == 'CC':
        lowerp = phat-k*np.sqrt(phat*(1-phat)/n)-1/(2*n)
        upperp = phat-k*np.sqrt(phat*(1-phat)/n)+1/(2*n)
    elif method == 'CWS':
        lowerp = (2*n*phat+k**2-1-k*np.sqrt(k**2-2-(1/n)+4*phat*(n*(1-phat)+1)))/(2*(n+k**2))
        upperp = (2*n*phat+k**2+1+k*np.sqrt(k**2+2-(1/n)+4*phat*(n*(1-phat)-1)))/(2*(n+k**2))
    lowerp = max(0,lowerp)
    upperp = min(upperp,1)
    lower = scipy.stats.binom.ppf(1-P, n = m, p = lowerp)
    upper = scipy.stats.binom.ppf(P, n = m, p = upperp)
    if side == 2:
        alpha = 2*alpha
        P = (2*P)-1
        temp = pandas.DataFrame([[alpha,P, phat,lower,upper]],columns=['alpha','P','p.hat','2-sided.lower','2-sided.upper'])
    else:
        temp = pandas.DataFrame([[alpha,P, phat,lower,upper]],columns=['alpha','P','p.hat','1-sided.lower','1-sided.upper'])   
    return temp

