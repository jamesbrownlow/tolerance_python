import pandas as pd
import numpy as np
import scipy.stats

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
        
        exp2tol.int(x = x, alpha = 0.05, P = 0.90, side = 1, method = "DUN", type.2 = FALSE)
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
        r = n - x.count(mx)
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

def paretotolint(x, alpha = 0.05, P = 0.99, side = 1, method = 'GPU', powerdist = False):
    '''
    Pareto (or Power Distribution) Tolerance Intervals

Description
    Provides 1-sided or 2-sided tolerance intervals for data distributed 
    according to either a Pareto distribution or a power distribution 
    (i.e., the inverse Pareto distribution).
    
    paretotolint(x, alpha = 0.05, P = 0.99, side = 1,
              method = ["GPU", "DUN"], powerdist = FALSE)
        
Parameters
----------
    x: list
        A vector of data which is distributed according to either a Pareto 
        distribution or a power distribution.
    
    alpha: float, optional
        The level chosen such that 1-alpha is the confidence level. The 
        default is 0.05.
    
    P: float, optional
        The proportion of the population to be covered by this tolerance 
        interval. The default is 0.99.
    
    side: 1 or 2, optional	
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively). The default is 1.

    method: string, optional	
        The method for how the upper tolerance bound is approximated when 
        transforming to utilize the relationship with the 2-parameter 
        exponential distribution. "GPU" is the Guenther-Patil-Upppuluri method.
        "DUN" is the Dunsmore method, which was empirically shown to be an 
        improvement for samples greater than or equal to 8. More information 
        on these methods can be found in the "References". The default is 
        "GPU"

    powerdist, bool
        If True, then the data is considered to be from a power distribution, 
        in which case the output gives tolerance intervals for the power 
        distribution. The default is False.
    
Details
    Recall that if the random variable X is distributed according to a Pareto 
    distribution, then the random variable Y = ln(X) is distributed according 
    to a 2-parameter exponential distribution. Moreover, if the random 
    variable W is distributed according to a power distribution, then the 
    random variable X = 1/W is distributed according to a Pareto distribution, 
    which in turn means that the random variable Y = ln(1/W) is distributed 
    according to a 2-parameter exponential distribution.
    
Returns
-------
    paretotol.int returns a data frame with items:
        alpha:	
            The specified significance level.

        P:	
            The proportion of the population covered by this tolerance 
            interval.

        1-sided.lower:
            The 1-sided lower tolerance bound. This is given only if side = 1.

        1-sided.upper:	
            The 1-sided upper tolerance bound. This is given only if side = 1.

        2-sided.lower:	
            The 2-sided lower tolerance bound. This is given only if side = 2.

        2-sided.upper:	
            The 2-sided upper tolerance bound. This is given only if side = 2.



References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Dunsmore, I. R. (1978), Some Approximations for Tolerance Factors for the 
        Two Parameter Exponential Distribution, Technometrics, 20, 317–318.

    Engelhardt, M. and Bain, L. J. (1978), Tolerance Limits and Confidence 
        Limits on Reliability for the Two-Parameter Exponential Distribution, 
        Technometrics, 20, 37–39.

    Guenther, W. C., Patil, S. A., and Uppuluri, V. R. R. (1976), One-Sided 
        β-Content Tolerance Factors for the Two Parameter Exponential 
        Distribution, Technometrics, 18, 333–340.

    Krishnamoorthy, K., Mathew, T., and Mukherjee, S. (2008), Normal-Based 
        Methods for a Gamma Distribution: Prediction and Tolerance Intervals 
        and Stress-Strength Reliability, Technometrics, 50, 69–78.
    
Examples
--------
    ## 95%/99% 2-sided Pareto tolerance intervals for a sample of size 500. 
        
        x = np.random.exponential(size=500)
        
        paretotolint(x = x, alpha = 0.05, P = 0.99, side = 2, method = "DUN", power.dist = FALSE)
    '''
    if side != 1 and side != 2:
        return "Must specify a one-sided or two-sided procedure!"
    if powerdist:
        x = np.log(1/x)
    else:
        x = np.log(x)
    out = exp2tolint(x = x, alpha = alpha, P = P, side = side, method = method, type2 = False)
    if side == 1:
        lower = out['1-sided.lower']
        upper = out['1-sided.upper']
    else:
        lower = out['2-sided.lower']
        upper = out['2-sided.upper']
    if powerdist:
        lower1 = 1/np.exp(upper)
        upper = 1/np.exp(lower)
        lower = lower1
    else:
        lower = np.exp(lower)
        upper = np.exp(upper)
    if side == 2:
        return pd.DataFrame({'alpha':[alpha],'P':[P],'2-sided.lower':lower,'2-sided.upper':upper})
    else:
        return pd.DataFrame({'alpha':[alpha],'P':[P],'1-sided.lower':lower,'1-sided.upper':upper})
