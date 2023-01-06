import numpy as np
import pandas as pd
import scipy.stats as st

def length(x):
    if type(x) == int or type(x) == float:
        return 1
    else:
        return len(x)

def umatolint(x, N = None, n = None, dist = 'Bin', alpha = 0.05, P = 0.99):
    '''
Uniformly Most Accurate Upper Tolerance Limits for Certain Discrete 
Distributions

Description
-----------
    Provides uniformly most accurate upper tolerance limits for the binomial, 
    negative binomial, and Poisson distributions.
    
    umatolint(x, n = NULL, dist = ["Bin", "NegBin", "Pois"], N, alpha = 0.05, 
              P = 0.99)

Parameters
----------
    x : float or list of floats
        A vector of data which is distributed according to one of the 
        binomial, negative binomial, or Poisson distributions. If the 
        length of x is 1,then it is assumed that this number is the sum of 
        iid values from the assumed distribution.
        
    N : int, optional
        Must be specified for the binomial and negative binomial distributions.
        If dist = "Bin", then N is the number of Bernoulli trials and must be 
        a positive integer. If dist = "NegBin", then N is the total number of 
        successful trials (or dispersion parameter) and must be strictly 
        positive. The default is None.
        
    n : int, optional
        The sample size of the data. If None, then n is calculated as the 
        length of x. The default is None.
        
    dist : string, optional
        The distribution for the data given by x. The options are "Bin" for 
        the binomial distribution, "NegBin" for the negative binomial 
        distribution, and "Pois" for the Poisson distribution. The default is 
        'Bin'.
        
    alpha : float, optional
        The level chosen such that 1-alpha is the confidence level. The 
        default is 0.05.
        
    P : float, optional
        The proportion of the population to be covered by this tolerance 
        interval. The default is 0.99.

Returns
-------
    umatolint returns a data frame with items:

        alpha:	
            The specified significance level.

        P	
            The proportion of the population covered by this tolerance 
            interval.

        p.hat	
            The maximum likelihood estimate for the probability of success in 
            each trial; reported if dist = "Bin".

        nu.hat	
            The maximum likelihood estimate for the probability of success in 
            each trial; reported if dist = "NegBin".

        lambda.hat	
            The maximum likelihood estimate for the rate of success; reported 
            if dist = "Pois".

        1-sided.upper	
            The 1-sided upper tolerance limit.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Zacks, S. (1970), Uniformly Most Accurate Tolerance Limits for Monotone 
        Likelihood Ratio Families of Discrete Distributions, Journal of the 
        American Statistical Association, 65, 307–316.
        
Examples
--------
    ## Examples from Zacks (1970).

        umatolint(25, n = 4, dist = "Bin", N = 10, alpha = 0.10, 
           P = 0.95)
        
        umatolint(13, n = 10, dist = "NegBin", N = 2, alpha = 0.10,
           P = 0.95)
        
        umatolint(37, n = 10, dist = "Pois", alpha = 0.10, P = 0.95)
    '''
    if length(x) == 1 and n == None:
        return 'A value for n must be specified!'
    if length(x) > 1:
        n = len(x)
        y = sum(x)
    else:
        y = x    
    if dist == 'Bin' or dist == 'NegBin':
        if N == None:
            print('N must be specfied for the binomial and negative binomial distribution.',
                    'For the Poisson distribution, N doesn\'t need to be specified.')
            return ''
    if dist == 'Bin':
        if y > 0:
            r0 = 1-st.beta.ppf(alpha, N*n-y+1,y)
            r1 = 1-st.beta.ppf(alpha, N*n-y,y+1)
            R = max(r0,r1)
        else:
            R = 1-alpha**(1/(N*n))
        def f2(k, P, N):
            return st.beta.ppf(1-P,k+1,N-k)
        k = -1
        temp = -1
        while temp < R:
            k = k+1
            if k < N:
                temp = f2(k=k,P=P,N=N)
            else:
                temp = 1
        phat = (y/n)/N
        return pd.DataFrame({'alpha':[alpha],'P':[P],'p.hat':[phat],'1-sided.upper':k})
    if dist == 'NegBin':
        if y > 0:
            r1 = 1 - st.beta.ppf(alpha,n*N,y+1)
            r0 = 1 - st.beta.ppf(alpha,n*N,y)
            R = max(r0,r1)
        else:
            R = 1-alpha**(1/(n*N))
        k = -1
        temp = 1.1
        while temp > 1-R:
            k = k+1
            temp = st.beta.ppf(P,N,k+1)
        nuhat = N/(N+(y/n))
        return pd.DataFrame({'alpha':[alpha],'P':[P],'nu.hat':[nuhat],'1-sided.upper':k})
    if dist == 'Pois':
        if y > 0:
            r0 = st.chi2.ppf(1-alpha,2*y+2)/(2*n)
            r1 = st.chi2.ppf(1-alpha,2*y)/(2*n)
            R = 2*max(r0,r1)
        else:
            R = -(1/n)*np.log(alpha)
        k = -1
        temp = -1
        while temp < R:
            k = k+1
            temp = st.chi2.ppf(1-P,2*k+2)
        lambdahat = y/n
        return pd.DataFrame({'alpha':[alpha],'P':[P],'nu.hat':[lambdahat],'1-sided.upper':k})
    
umato