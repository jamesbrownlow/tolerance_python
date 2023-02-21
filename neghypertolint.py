import scipy.stats as st
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    return len(x)

def dnhyper(x,m,n,k,log=False):
    '''
The Negative Hypergeometric Distribution

Description
    Density function for the negative hypergeometric distribution.

Usage
    dnhyper(x, m, n, k, log = FALSE)

Parameters
----------
    x : list
        Vector of quantiles representing the number of trials until k 
        successes have occurred (e.g., until k white balls have been drawn 
        from an urn without replacement).
        
    m : int
        The number of successes in the population (e.g., the number of white 
        balls in the urn).
        
    n : int
        The population size (e.g., the total number of balls in the urn).
        
    k : int
        The number of successes (e.g., white balls) to achieve with the sample.
        
    log : bool, optional
        Logical vector. If True, then probabilities are given as log(p). 
        The default is False.

Details
    A negative hypergeometric distribution (sometimes called the inverse
    hypergeometric distribution) models the total number of trials until k 
    successes occur. Compare this to the negative binomial distribution, which 
    models the number of failures that occur until a specified number of 
    successes has been reached. The negative hypergeometric distribution has 
    density:
        
        p(x) = choose(x-1, k-1)choose(n-x, m-k) / choose(n, m)

    for x=k,k+1,...,n-m+k.

Returns
-------
    dnhyper gives the density.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Wilks, S. S. (1963), Mathematical Statistics, Wiley.
    
Examples
--------
    ## Randomly generated data from the negative hypergeometric distribution.
        x = rnhyper(nn = 1000, m = 15, n = 40, k = 10)
        
        x = sorted(x)
        
        dnhyper(x = x, m = 15, n = 40, k = 10)
    '''
    if k > m:
        return 'k cannot be larger than m.'
    p = []
    if length(x) > 1:
        for i in range(length(x)):
            p.append(st.hypergeom.pmf(k = k-1, M = n, N = x[i]-1, n = m)*(m-(k-1))/(n-(x[i]-1)))
    else:
        return st.hypergeom.pmf(k = k-1, M = n, N = x-1, n = m)*(m-(k-1))/(n-(x-1))
    for i in range(len(p)):
        if np.isnan(p[i]) or p[i] == None:
            p[i] = 0
    p = [min(max(val,0),1) for val in p]
    if log:
        p = np.log(p)
    return p

def pnhyper(q, m, n, k, lowertail = True, logp = False):
    '''
The Negative Hypergeometric Distribution

Description
    Distribution function for the negative hypergeometric distribution.

Usage
    pnhyper(q, m, n, k, lowertail = TRUE, logp = FALSE)   

Parameters
----------
    q : list
        Vector of quantiles representing the number of trials until k 
        successes have occurred (e.g., until k white balls have been drawn 
        from an urn without replacement).
        
    m : int
        The number of successes in the population (e.g., the number of white 
        balls in the urn).
        
    n : int
        The population size (e.g., the total number of balls in the urn).
        
    k : int
        The number of successes (e.g., white balls) to achieve with the sample.
        
    logp : bool, optional
        Logical vector. If True, then probabilities are given as log(p). 
        The default is False.
        
    lowertail : bool, optional
        Logical vector. If True, then probabilities are P[X≤ x], else P[X>x].
        The default is True. 

Details
    A negative hypergeometric distribution (sometimes called the inverse
    hypergeometric distribution) models the total number of trials until k 
    successes occur. Compare this to the negative binomial distribution, which 
    models the number of failures that occur until a specified number of 
    successes has been reached. The negative hypergeometric distribution has 
    density:
        
        p(x) = choose(x-1, k-1)choose(n-x, m-k) / choose(n, m)

    for x=k,k+1,...,n-m+k.

Returns
-------
    pnhyper gives the distribution function.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Wilks, S. S. (1963), Mathematical Statistics, Wiley.
    
Examples
--------
    ## Randomly generated data from the negative hypergeometric distribution.
        x = rnhyper(nn = 1000, m = 15, n = 40, k = 10)
        
        x = sorted(x)
        
        pnhyper(q = x, m = 15, n = 40, k = 10)
    '''
    if k > m:
        return 'k cannot be larger than m.'
    # == phyper(q = k-1, m = m, n = n-m, k = q, lower.tail = !lower.tail, log.p = log.p)
    if logp and lowertail:
        return st.hypergeom.logsf(k = k-1, M = n, N = q, n = m)
    elif (not logp) and lowertail:
        return st.hypergeom.sf(k = k-1, M = n, N = q, n = m) 
    elif logp and (not lowertail):
        return st.hypergeom.logcdf(k = k-1, M = n, N = q, n = m) 
    else:
        return st.hypergeom.cdf(k = k-1, M = n, N = q, n = m) 

def qnhyper(p, m, n, k, lowertail = True, logp = False):
    '''
The Negative Hypergeometric Distribution

Description
    Quantile function for the negative hypergeometric distribution.

Usage
    qnhyper(p, m, n, k, lowertail = TRUE, logp = FALSE)  

Parameters
----------
    p : flaot or list
        Vector of probabilities, which must be between 0 and 1.
        
    m : int
        The number of successes in the population (e.g., the number of white 
        balls in the urn).
        
    n : int
        The population size (e.g., the total number of balls in the urn).
        
    k : int
        The number of successes (e.g., white balls) to achieve with the sample.
        
    logp : bool, optional
        Logical vector. If True, then probabilities are given as log(p). 
        The default is False.
        
    lowertail : bool, optional
        Logical vector. If True, then probabilities are P[X≤ x], else P[X>x].
        The default is True. 

Details
    A negative hypergeometric distribution (sometimes called the inverse
    hypergeometric distribution) models the total number of trials until k 
    successes occur. Compare this to the negative binomial distribution, which 
    models the number of failures that occur until a specified number of 
    successes has been reached. The negative hypergeometric distribution has 
    density:
        
        p(x) = choose(x-1, k-1)choose(n-x, m-k) / choose(n, m)

    for x=k,k+1,...,n-m+k.

Returns
-------
    qnhyper gives the quantile function.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Wilks, S. S. (1963), Mathematical Statistics, Wiley.
    
Examples
--------
        qnhyper(p = 0.80, m = 15, n = 40, k = 10)
    '''
    if k > m:
        return 'k cannot be larger than m.'
    if logp:
        p = np.exp(p)
    allp = []
    temp = np.array(range(k,n-m+k+1))
    if lowertail:
        tmp1 = pd.DataFrame([k,temp[0]])
        tmp2 = pd.DataFrame([-np.inf,pnhyper(temp[0],m=m,n=n,k=k)])
        for i in range(n-m+k-(k-1)-1):
            tmp1.loc[len(tmp1.index)] = [temp[i+1]]
            tmp2.loc[len(tmp2.index)] = [pnhyper(temp[i+1],m=m,n=n,k=k)]
        tmp1.loc[len(tmp1.index)] = n-m+k
        tmp2.loc[len(tmp2.index)] = np.inf
        tempout = pd.concat([tmp1,tmp2],axis = 1)
        tempout.columns = ['temp','']
    else:
        tmp1 = pd.DataFrame([k,temp[0]])
        tmp2 = pd.DataFrame([np.inf,pnhyper(temp[0],m=m,n=n,k=k,lowertail = False)])
        for i in range(n-m+k-(k-1)-1):
            tmp1.loc[len(tmp1.index)] = [temp[i+1]]
            tmp2.loc[len(tmp2.index)] = [pnhyper(temp[i+1],m=m,n=n,k=k,lowertail = False)]
        tmp1.loc[len(tmp1.index)] = n-m+k
        tmp2.loc[len(tmp2.index)] = -np.inf
        tempout = pd.concat([tmp1,tmp2],axis = 1)
        tempout.columns = ['temp','']
    if length(p) > 1:
        for i in range(length(p)):
            if lowertail:
                allp.append(min(tempout[tempout['']>=p[i]]['temp']))
            else:
                allp.append(min(tempout[tempout['']<p[i]]['temp']))
    else:
        if lowertail:
                allp.append(min(tempout[tempout['']>=p]['temp']))
        else:
                allp.append(min(tempout[tempout['']<p]['temp']))
    return allp

def rnhyper(nn, m, n, k):
    '''
The Negative Hypergeometric Distribution

Description
    Random generation for the negative hypergeometric distribution.

Usage
    rnhyper(nn, m, n, k)   

Parameters
----------
    nn : int
        The number of observations. If length > 1, then the length is taken
        to be the number required. 
        
    m : int
        The number of successes in the population (e.g., the number of white 
        balls in the urn).
        
    n : int
        The population size (e.g., the total number of balls in the urn).
        
    k : int
        The number of successes (e.g., white balls) to achieve with the sample.

Details
    A negative hypergeometric distribution (sometimes called the inverse
    hypergeometric distribution) models the total number of trials until k 
    successes occur. Compare this to the negative binomial distribution, which 
    models the number of failures that occur until a specified number of 
    successes has been reached. The negative hypergeometric distribution has 
    density:
        
        p(x) = choose(x-1, k-1)choose(n-x, m-k) / choose(n, m)

    for x=k,k+1,...,n-m+k.

Returns
-------
    rnhyper generates random deviates.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Wilks, S. S. (1963), Mathematical Statistics, Wiley.
    
Examples
--------
    ## Randomly generated data from the negative hypergeometric distribution.
        x = rnhyper(nn = 1000, m = 15, n = 40, k = 10)
    '''
    if k > m:
        return 'k cannot be larger than m.'
    return qnhyper(np.random.uniform(size = nn),m,n,k)

def neghypertolint(x, n, N, m = None, alpha = 0.05, P = 0.99, side = 1, method = 'EX'):
    '''    
Negative Hypergeometric Tolerance Intervals

Description
    Provides 1-sided or 2-sided tolerance intervals for negative 
    hypergeometric random variables. When sampling without replacement, these 
    limits are on the total number of expected draws in a future sample in 
    order to achieve a certain number from group A (e.g., "black balls" in an 
    urn).

Usage
    neghypertol.int(x, n, N, m = NULL, alpha = 0.05, P = 0.99,
                    side = 1, method = c("EX", "LS", "CC"))

Parameters
----------
    x : int or list
        The number of units drawn in order to achieve n successes. Can be a 
        vector, in which case the sum of x is used.

    n : int
        The target number of successes in the sample drawn (e.g., the number 
        of "black balls" you are to draw in the sample).
        
    N : int
        The population size (e.g., the total number of balls in the urn).
        
    m : int, optional
       The target number of successes to be sampled from the universe for a 
       future study. If m = NULL, then the tolerance limits will be 
       constructed assuming n for this quantity. The default is None.
       
    alpha : float, optional
        The level chosen such that 1-alpha is the confidence level. The 
        default is 0.05.
        
    P : TYPE, optional
        The proportion of units from group A in future samples of size m to be
        covered by this tolerance interval. The default is 0.99.
        
    side : 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively).. The default is 1.
        
    method : string, optional
        The method for calculating the lower and upper confidence bounds, 
        which are used in the calculation of the tolerance bounds. 
            The default method is "EX", which is an exact-based method. 
            
            "LS" is the large-sample method. 
            
            "CC" gives a continuity-corrected version of the large-sample 
            method.

Returns
-------
    neghypertol.int returns a data frame with items:

        alpha:	
            The specified significance level.

        P:	
            The proportion of units from group A in future samples of size m.

        rate:	
            The sampling rate determined by x/N.

        p.hat:
            The proportion of units in the sample from group A, calculated by n/x.

        1-sided.lower:	
            The 1-sided lower tolerance bound. This is given only if side = 1.

        1-sided.upper:	
            The 1-sided upper tolerance bound. This is given only if side = 1.

        2-sided.lower:	
            The 2-sided lower tolerance bound. This is given only if side = 2.

        2-sided.upper:	
            The 2-sided upper tolerance bound. This is given only if side = 2.
            
Note
    As this methodology is built using large-sample theory, if the sampling 
    rate is less than 0.05, then a warning is generated stating that the
    results are not reliable.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.

    Khan, R. A. (1994), A Note on the Generating Function of a Negative 
        Hypergeometric Distribution, Sankhya: The Indian Journal of Statistics,
        Series B, 56, 309–313.

    Young, D. S. (2014), Tolerance Intervals for Hypergeometric and Negative 
        Hypergeometric Variables, Sankhya: The Indian Journal of Statistics, 
        Series B, 77(1), 114–140.

Examples
--------
    ## 90%/95% 2-sided negative hypergeometric tolerance intervals for a 
    future number of 20 successes when the universe is of size 100.  The 
    estimates are based on having drawn 50 in another sample to achieve 20 
    successes.

        neghypertol.int(50, 20, 100, m = 20, alpha = 0.05, 
                        P = 0.95, side = 2, method = "LS")
    '''
    if side != 1 and side != 2:
        return 'Must specify a one-sided or two-sided procedure'
    if side == 2:
        alpha = alpha/2
        P = (P+1)/2
    if length(x) > 1:
        x = sum(x)
    rate = x/N
    if rate < 0.05:
        print('Warning: Sampling rate < 0.05. Results may not be accurate.')
    nuhat = n/x
    k = st.norm.ppf(1-alpha)
    if m == None:
        m = n
    if (nuhat**2*(1-nuhat))/x < 0 or (N-x)/(N-1) < 0:
        print("Warning message:\nNaNs produced, sqrt of negative number was taken. ")
    fpc = np.sqrt((N-x)/(N-1))
    senuhat = np.sqrt((nuhat**2*(1-nuhat))/x)*fpc
    if method == 'EX':
        temp = list(range(n,N-(x-n)+1))
        lowerex = []
        upperex = []
        for i in range(length(temp)):
            lowerex.append(round(pnhyper(x,temp[i],N,n),9))
            upperex.append(round(1-pnhyper(x-1,temp[i],N,n),9))
        try:
            lowermin = min(np.where(np.array(lowerex)>alpha)[0])
            uppermax = max(np.where(np.array(upperex)>alpha)[0])
        except:
            return"Error message:\n NaNs produced."
        Ml = temp[int(min(length(temp),lowermin))]
        Mu = temp[int(max(1,uppermax))]
    
    if method == 'LS' or method == 'CC':
        lowerp = max(0.0000001, nuhat-k*senuhat-1/(2*x)*(method == 'CC'))
        upperp = min(1, nuhat+k*senuhat+1/(2*x)*(method == 'CC'))
        lowerp = max(0,lowerp)
        upperp = min(upperp,1)
        Mu = int(max(m, min(np.ceil(N*upperp),N)))
        Ml = int(max(m, np.floor(N*lowerp)))
    
    lower = qnhyper(1-P, m = Mu, n = N, k = m)
    upper = qnhyper(P, m = Ml, n = N, k = m)  
    if side == 2:
        alpha = 2* alpha
        P = (2*P)-1
        return pd.DataFrame({'alpha':[alpha],'P':[P],'rate':[rate],'p.hat':[nuhat],'2-sided.lower':lower,'2-sided.upper':upper})
    else:
        return pd.DataFrame({'alpha':[alpha],'P':[P],'rate':[rate],'p.hat':[nuhat],'1-sided.lower':lower,'1-sided.upper':upper})
        
# print(neghypertolint(50, n=23, N=154, method = 'LS', side = 1))
# print(neghypertolint(50, n=20, N=100, m = 20, method = "LS", side = 1))
# print(neghypertolint(50, n=23, N=154, method = 'EX', side = 1))
# print(neghypertolint(50, n=20, N=100, m = 20, method = "EX", side = 1))
# print(neghypertolint(50, n=23, N=154, method = 'CC', side = 1))
# print(neghypertolint(50, n=20, N=100, m = 20, method = "CC", side = 2))
# print(neghypertolint(50, n=23, N=154, method = 'LS', side = 2))
# print(neghypertolint(50, n=20, N=100, m = 20, method = "LS", side = 2))
# print(neghypertolint(50, n=23, N=154, method = 'EX', side = 2))
# print(neghypertolint(50, n=20, N=100, m = 20, method = "EX", side = 2))
# print(neghypertolint(50, n=23, N=154, method = 'CC', side = 2))
# print(neghypertolint(50, n=20, N=100, m = 20, method = "CC", side = 2))