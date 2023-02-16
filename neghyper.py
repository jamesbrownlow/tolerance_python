import scipy.stats as st
import numpy as np
import pandas as pd

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
    if length(allp) == 1:
        try:
            return allp[0]
        except:
            return allp
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

#x = rnhyper(nn = 1000, m = 15, n = 40, k = 10)
# #x = sorted(x)
# x = [.01,.1,.5,.7,.9,.99]
# print(dnhyper(x = x, m = 15, n = 40, k = 10))

# #x = rnhyper(nn = 1000, m = 15, n = 40, k = 10)
# #x = sorted(x)

# print(pnhyper(q = x, m = 15, n = 40, k = 10))
# print(qnhyper(p = 0.80, m = 15, n = 40, k = 10))
# x = rnhyper(nn = 1000, m = 15, n = 40, k = 10)
# print(x)

#print(qnhyper(p = [0.8,0.9], m = 15, n = 40, k = 10,lowertail=True))
#print(rnhyper(nn=10, m = 15, n = 40, k = 10))
#x = rnhyper(nn=10, m = 15, n = 40, k = 10)
#print(dnhyper(x=[31,20,20,27,26,26,30,25,24,29],m=15,n=40,k=10))
#print(dnhyper(x=25,m=15,n=40,k=10))