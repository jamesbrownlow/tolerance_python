import pandas as pd
import numpy as np
import scipy.stats
import statistics as st

def length(x):
    if type(x) == int or type(x) == float:
        return 1
    else:
        return len(x)
    
def diffnormtolint(x1, x2, varratio = None, alpha = 0.05, P = 0.99, method = "HALL"):
    '''
1-Sided Tolerance Limits for the Distribution of the Difference Between Two 
Independent Normal Random Variables

Description
    Provides 1-sided tolerance limits for the difference between two 
    independent normal random variables. If the ratio of the variances is 
    known, then an exact calculation is performed. Otherwise, approximation 
    methods are implemented.
    
    diffnormtolint(x1, x2, varratio = None, alpha = 0.05, P = 0.99, method = ["HALL","GK","RG"])
        
Parameters
----------
    x1: list
        A vector of sample data which is distributed according to a normal 
        distribution (sample 1).

    x2: list
       Another vector of sample data which is distributed according to a 
       normal distribution (sample 2). It can be of a different sample size 
       than the sample specified by x1.
      
    varratio: float
        A specified, known value of the variance ratio (i.e., the ratio of 
        the variance for population 1 to the variance of population 2). If 
        NULL, then the variance ratio is estimated according to one of the 
        three methods specified in the method argument. The default value 
        is None. 
    
    alpha: float, optional
        The level chosen such that 1-alpha is the confidence level.
        The default is 0.05.
    
    P: float, optional
        The proportion of the population to be covered by this tolerance 
        interval. The default is 0.99.
    
    method: string, optional
        The method for estimating the variance ratio. This only needs to be 
        specified in the case when var.ratio is not NULL. 
        
            "HALL" is Hall's method, which takes a bias-corrected version of 
            the ratio between the sample variance for sample 1 to the sample 
            variance for sample 2. 
            
            "GK" is the Guo-Krishnamoorthy method, which first calculates a 
            bias-corrected version of the ratio between the sample variance 
            for sample 2 to the sample variance for sample 1. The resulting 
            limit is then compared to the limit from Hall's method and the 
            most conservative limit is chosen. 
            
            "RG" is the Reiser-Guttman method, which is a biased version of 
            the variance ratio that is calculated by taking the sample 
            variance for sample 1 to the sample variance for sample 2. 
            Typically, Hall's method or the Guo-Krishnamoorthy method are 
            preferred to the Reiser-Guttman method.
        
        The default is 'HALL'.
    
Details
    Satterthwaite's approximation for the degrees of freedom is used when the 
    variance ratio is unknown.
    
Returns
-------
  diffnormtolint returns a data frame with items:
        
    alpha: 
        The specified significance level.
    P: 
        The proportion of the population covered by this tolerance interval.
    diff.bar:
        The difference between the sample means.
    1-sided.lower: 
        The 1-sided lower tolerance bound.
    1-sided.upper: 
        The 1-sided upper tolerance bound.

References
---------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
    
    Guo, H. and Krishnamoorthy, K. (2004), New Approximate Inferential 
        Methods for the Reliability Parameter in a Stress-Strength Model: 
        The Normal Case, Communications in Statistics - Theory and Methods, 
        33, 1715–1731.

    Hall, I. J. (1984), Approximate One-Sided Tolerance Limits for the 
        Difference or Sum of Two Independent Normal Variates, Journal of 
        Quality Technology, 16, 15–19.

    Krishnamoorthy, K. and Mathew, T. (2009), Statistical Tolerance Regions: 
        Theory, Applications, and Computation, Wiley.

    Reiser, B. J. and Guttman, I. (1986), Statistical Inference for Pr(Y < X): 
        The Normal Case, Technometrics, 28, 253–257.
        
Examples
--------
## 90%/99% tolerance limits for the difference between two simulated normal 
data sets.
    x1 = np.random.normal(size = 100, loc = 1, scale = 10)
    
    x2 = np.random.normal(size = 20, loc = 50, scale = 1)
    
    diffnormtolint(x1, x2, alpha = 0.10, varratio = 10)


    x1 = [10.166, 5.889, 8.258, 7.303, 8.757]
    
    x2 = [-0.204, 2.578, 1.182, 1.892, 0.786, -0.517, 1.156,
        0.980, 0.323, 0.437, 0.397, 0.050, 0.812, 0.720]
    
    diffnormtolint(x1, x2, alpha = 0.10, method = 'RG', varratio = None)
    '''
    n1 = length(x1)
    n2 = length(x2)
    x1bar = np.mean(x1)
    x2bar = np.mean(x2)
    s1_2 = st.variance(x1)
    s2_2 = st.variance(x2)
    zp = scipy.stats.norm.ppf(P)
    if varratio == None:
        if method == "HALL" or method == 'GK':
            q1 = (s1_2 * (n2 - 3))/(s2_2 * (n2 - 1))
            q2 = (s2_2 * (n1 - 3))/(s1_2 * (n1 - 1))
            f1 = ((n1 - 1) * (q1 + 1)**2)/(q1**2 + (n1 - 1)/(n2 -1))
            f2 = ((n2 - 1) * (q2 + 1)**2)/(q2**2 + (n2 - 1)/(n1 -1))
            nu1 = (n1 * (1 + q1))/(q1 + (n1/n2))
            nu2 = (n2 * (1 + q2))/(q2 + (n2/n1))
            lower = x1bar - x2bar - scipy.stats.nct.ppf(1 - alpha, f1, nc = (zp * np.sqrt(nu1))) * np.sqrt((s1_2 + s2_2)/nu1)
            upper = x1bar - x2bar + scipy.stats.nct.ppf(1 - alpha, f1, nc = (zp * np.sqrt(nu1))) * np.sqrt((s1_2 + s2_2)/nu1)
            if method == 'GK':
                loweralt = x1bar - x2bar - scipy.stats.nct.ppf(1 - alpha, f2, nc = (zp * np.sqrt(nu2))) * np.sqrt((s1_2 + s2_2)/nu2)
                upperalt = x1bar - x2bar + scipy.stats.nct.ppf(1 - alpha, f2, nc = (zp * np.sqrt(nu2))) * np.sqrt((s1_2 + s2_2)/nu2)
                lower = min(lower,loweralt)
                upper = max(upper,upperalt)
        elif method == 'RG':
            q1 = s1_2/s2_2
            f1 = ((n1 - 1) * (q1 + 1)**2)/(q1**2 + (n1 - 1)/(n2 -1))
            nu1 = (n1 * (1 + q1))/(q1 + (n1/n2))
            lower = x1bar - x2bar - scipy.stats.nct.ppf(1 - alpha, f1, nc = (zp * np.sqrt(nu1))) * np.sqrt((s1_2 + s2_2)/nu1)
            upper = x1bar - x2bar + scipy.stats.nct.ppf(1 - alpha, f1, nc = (zp * np.sqrt(nu1))) * np.sqrt((s1_2 + s2_2)/nu1)
    else:
        q1 = varratio
        nu1 = (n1 * (1 + q1))/(q1 + (n1/n2))
        sd = np.sqrt(((1 + 1/q1) * ((n1 - 1) * s1_2 + (n2 - 1) * q1 * s2_2))/(n1 + n2 - 2))
        lower = x1bar - x2bar - scipy.stats.nct.ppf(1 - alpha, n1 + n2 - 2, nc = (zp * np.sqrt(nu1))) * sd/np.sqrt(nu1)
        upper = x1bar - x2bar + scipy.stats.nct.ppf(1 - alpha, n1 + n2 - 2, nc = (zp * np.sqrt(nu1))) * sd/np.sqrt(nu1)
    return pd.DataFrame({"alpha":[alpha], "P":[P], "diff.bar":[x1bar-x2bar], "1-sided.lower":lower, "1-sided.upper":upper})

#x1 = np.random.normal(size = 100, loc = 1, scale = 10)
#x2 = np.random.normal(size = 20, loc = 50, scale = 1)

#x1 = [1,2,3,4,5,6,7,8,9,9,9,9,8,8,7,6,5,4,4,4,3,3,2,1,1,1,8,8,2,9,9]
#x2 = [20, 40, 70, 90, 10]
#print(diffnormtolint(x1, x2,method = 'RG',varratio = None))