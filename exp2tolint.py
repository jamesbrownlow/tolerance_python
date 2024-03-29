import numpy as np
import scipy.stats
import pandas as pd

def exp2tolint(x,alpha=0.05,P=0.99,side=1,method = "GPU", type2 = False):
    '''
    exp2tolint(x, alpha = 0.05, P = 0.99, side = 1,method = c("GPU", "DUN", "KM"), type.2 = FALSE)
    
2-Parameter Exponential Tolerance Intervals

Description:
    Provides 1-sided or 2-sided tolerance intervals for data distributed according to a 2-parameter exponential distribution. Data with Type II censoring is permitted.

Parameters
----------
    x : list
        A vector of data which is distributed according to the 2-parameter 
        exponential distribution.
    alpha : flaot, optional
        The level chosen such that 1-alpha is the confidence level.
        The default is 0.05.
    P : flaot, optional
        The proportion of the population to be covered by this tolerance 
        interval. The default is 0.99.
    side : 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively). The default is 1.
    method : string, optional
        The method for how the upper tolerance bound is approximated. "GPU" 
        is the Guenther-Patil-Upppuluri method. "DUN" is the Dunsmore method, 
        which has been empirically shown to be an improvement for samples 
        greater than or equal to 8. "KM" is the Krishnamoorthy-Mathew method,
        which is typically more liberal than the other methods. More 
        information on these methods can be found in the "References", which 
        also highlight general sample size conditions as to when these 
        different methods should be used. The default is "GPU".
    type2 : bool, optional
        Select TRUE if Type II censoring is present 
        (i.e., the data set is censored at the maximum value present). 
        The default is False.

Returns
-------
        Returns a dataframe with items:
            alpha:
                The specified significance level.
            P:
                The proportion of the population covered by this tolerance 
                interval.
            1-sided.lower:
                The 1-sided lower tolerance bound. 
                This is given only if side = 1.
            1-sided.upper:	
                The 1-sided upper tolerance bound. 
                This is given only if side = 1.
            2-sided.lower:
                The 2-sided lower tolerance bound. 
                This is given only if side = 2.
            2-sided.upper:
                The 2-sided upper tolerance bound. 
                This is given only if side = 2.
References
----------
        Derek S. Young (2010). tolerance: An R Package for Estimating 
            Tolerance Intervals. Journal of Statistical Software, 36(5), 
            1-39. URL http://www.jstatsoft.org/v36/i05/.
        Dunsmore, I. R. (1978), Some Approximations for Tolerance Factors 
            for the Two Parameter Exponential Distribution, Technometrics, 
            20, 317–318.

        Engelhardt, M. and Bain, L. J. (1978), Tolerance Limits and Confidence 
            Limits on Reliability for the Two-Parameter Exponential 
            Distribution, Technometrics, 20, 37–39.
    
        Guenther, W. C., Patil, S. A., and Uppuluri, V. R. R. (1976), 
            One-Sided β-Content Tolerance Factors for the Two Parameter 
            Exponential Distribution, Technometrics, 18, 333–340.

        Krishnamoorthy, K. and Mathew, T. (2009), Statistical Tolerance 
            Regions: Theory, Applications, and Computation, Wiley.
            
Examples
--------
    ## 95%/90% 1-sided 2-parameter exponential tolerance intervals for a sample of size 50. 
        x = np.random.exponential(226,size = 50)
        
        exp2tolint(x = x, alpha = 0.05, P = 0.90, side = 1, method = "DUN", type2 = False)
        
    '''
    if side != 1 and side != 2:
        return 'Must specify a one-sided or two-sided procedure'
    if side == 2:
        alpha = alpha/2
        P = (P+1)/2
    n = len(x)
    T = min(x)
    S = sum([y-T for y in x]) 
    if type2:
        mx = max(x)
        r = n - list(x).count(mx)
        def m(P,R,n):
            return (1+n*np.log(P))/(r-(5/2))
        k1 = (-m(P,r,n)-scipy.stats.norm.ppf(1-alpha)*np.sqrt(m(P,r,n)**2/r+(1/r**2)))/n
        k2 = (-m(1-P,r,n)-scipy.stats.norm.ppf(alpha)*np.sqrt(m(1-P,r,n)**2/r+(1/r**2)))/n
    else:
        k1 = (1-((P**n)/alpha)**(1/(n-1)))/n
        if method == 'KM':
            k2 = (1 - (((1-P)**n)/(1-alpha))**(1/(n-1)))/n
        else:
            k2 = scipy.stats.chi2.ppf(P,2)/scipy.stats.chi2.ppf(alpha,2*n-2)
            if method == 'DUN':
                lambda1 = 1.71+1.57*np.log(np.log(1/alpha))
                k2 = k2 -(lambda1/n)**(1.63+0.39*lambda1)
    lower = T+S*k1
    upper = T+S*k2
    if side == 2:
        alpha = 2*alpha
        P = (2*P)-1
        temp = pd.DataFrame({'alpha':[alpha],'P':[P],'2-sided.lower':[lower],'2-sided.upper':[upper]})
    else:
        temp = pd.DataFrame({'alpha':[alpha],'P':[P],'1-sided.lower':[lower],'1-sided.upper':[upper]})
    return temp

#x = [200.47317759 ,515.27543654 ,502.01117096 ,382.37767967  , 8.26017642,265.97233374 ,107.55578385 ,165.17702228, 242.45339628,  48.69977269]

# print(exp2tolint(x = x, alpha = 0.05, P = 0.9, side = 1, method = "DUN", type2 = True))
# print(exp2tolint(x = x, alpha = 0.05, P = 0.9, side = 1, method = "DUN", type2 = False))
# print(exp2tolint(x = x, alpha = 0.05, P = 0.9, side = 2, method = "DUN", type2 = True))
# print(exp2tolint(x = x, alpha = 0.05, P = 0.9, side = 2, method = "DUN", type2 = False))
# print(exp2tolint(x = x, alpha = 0.05, P = 0.9, side = 1, method = "KM", type2 = True))
# print(exp2tolint(x = x, alpha = 0.05, P = 0.9, side = 1, method = "KM", type2 = False))
# print(exp2tolint(x = x, alpha = 0.05, P = 0.9, side = 2, method = "KM", type2 = True))
# print(exp2tolint(x = x, alpha = 0.05, P = 0.9, side = 2, method = "KM", type2 = False))