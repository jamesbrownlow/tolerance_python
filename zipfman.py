import numpy as np
import scipy.stats as st
from scipy.special import zeta
import pandas as pd
from statsmodels.formula.api import ols



def length(x=None):
    try:
        return len(x)
    except:
        if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
            return 1
        else:
            return 0       

def dzipfman(x, s, b = None, N = None, log = False):
    '''
Zipf-Mandelbrot Distributions

Description
    Density (mass) function for the Zipf, Zipf-Mandelbrot, and zeta 
    distributions.

Usage
    dzipfman(x, s = None, b = None, N = None, log = False)
    
Parameters
----------
    x: list
        Vector of quantiles.

    s, b: float
        The shape parameters, both of which must be greater than 0. b must be 
        specified for Zipf-Mandelbrot distributions.

    N: int
        The number of categories, which must be integer-valued for Zipf and 
        Zipf-Mandelbrot distributions. For a zeta distribution, N = Inf must be used.

    log: bool
        Logical vectors. If TRUE, then the probabilities are given as log(p).

Details
-------

The Zipf-Mandelbrot distribution has mass

    p(x) = (x + b)^-s/∑_{i=1}^{N}(i + b)^(-s),

where x=1,…,N, s,b>0 are shape parameters, and N is the number of distinct 
categories. The Zipf distribution is just a special case of the 
Zipf-Mandelbrot distribution where the second shape parameter b=0. The zeta 
distribution has mass

p(x) = x^-λ/ζ(s),

where x=1,2,…, s>1 is the shape parameter, and ζ() is the Riemann zeta 
function given by:

ζ(t) = ∑_{i=1}^∞ 1/i^t<∞.

Note that the zeta distribution is just a special case of the Zipf 
distribution where s>1 and N goes to infinity.

Value
-----

    dzipfman gives the density (mass), pzipfman gives the distribution function, 
    qzipfman gives the quantile function, and rzipfman generates random deviates 
    for the specified distribution.

Note
----

    These functions may be updated in a future version of the package so as to 
    allow greater flexibility with the inputs.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Mandelbrot, B. B. (1965), Information Theory and Psycholinguistics. In B. 
        B. Wolman and E. Nagel, editors. Scientific Psychology, Basic Books.

    Young, D. S. (2013), Approximate Tolerance Limits for Zipf-Mandelbrot 
        Distributions, Physica A: Statistical Mechanics and its Applications, 
        392, 1702–1711.

    Zipf, G. K. (1949), Human Behavior and the Principle of Least Effort, 
        Hafner.

    Zornig, P. and Altmann, G. (1995), Unified Representation of Zipf 
        Distributions, Computational Statistics and Data Analysis, 19, 461–473.

Examples
--------
## Randomly generated data from the Zipf distribution.

    dzipfman(x = 1, s = 2, N = 100)

## Randomly generated data from the Zipf-Mandelbrot distribution.

    dzipfman(x = [4,5,9], s = 2, b = 3, N = 100)

## Randomly generated data from the zeta distribution.

    dzipfman(x = [4,8], s = 1.3, N = np.inf)

    '''
    if N == None:
        return 'Must specify N.'
    out = np.linspace(0,0,length(x))
    if length(x) == 1:
        x = [x]
    x = np.array(x)
    temp = [a if ((a != np.floor(a) and a <= N) or (a < 1) or (a == np.floor(a) and (a>N))) else 0 for a in x]
    if sum(temp) > 0:
        out = np.linspace(0,0,length(temp))
    if sum(temp) != length(x):
        if b == None and N < np.inf:
            if s <= 0:
                return "Invalid value for s!"
            temp = np.array(temp)
            out[np.where(out==temp)] = (np.float_power(x[np.where(x != temp)],-s))/sum(np.float_power(range(1,int(N+1)),-s))
        elif b != None:
            if s <= 0 or b < 0:
                return "Invalid value for s and/or b!"
            if N == np.inf:
                return "N must be finite!"
            out[np.where(out==temp)] = (np.float_power((x[np.where(x != temp)] + b),-s))/(sum(np.float_power((np.array(range(1,N+1)) + b),-s)))
        else:
            if s <= 1:
                return "Invalid value for s!"
            out[np.where(out==temp)] = (np.float_power(x[np.where(x != temp)],-s))/zeta(s)
    if log:
        out = np.log(out)
    if any(x != np.floor(x)):
        ind = np.where(x!=np.floor(x))[0]
        for i in range(length(ind)):
            if i == 0:
                print("Warning messages:")
            print(f'{i+1}: In the function dzipfman, non-integer x = {x[ind[i]]}')
            if i == length(ind)-1:
                print('\n')
    return out

def pzipfman(q, s, b = None, N = None, lowertail = True, logp = False):
    '''
    Zipf-Mandelbrot Distributions

Description
    distribution function for the Zipf, Zipf-Mandelbrot, and zeta 
    distributions.

Usage
    pzipfman(q, s, b = None, N = None, lowertail = True, logp = False)
    
Parameters
----------
    x: list
        Vector of quantiles.

    s, b: float
        The shape parameters, both of which must be greater than 0. b must be 
        specified for Zipf-Mandelbrot distributions.

    N: int
        The number of categories, which must be integer-valued for Zipf and 
        Zipf-Mandelbrot distributions. For a zeta distribution, N = Inf must be used.

    logp: bool
        Logical vectors. If TRUE, then the probabilities are given as log(p).

    lowertail: bool
        Logical vector. If TRUE, then probabilities are P[X≤ x], else P[X>x].

Details
-------

The Zipf-Mandelbrot distribution has mass

    p(x) = (x + b)^-s/∑_{i=1}^{N}(i + b)^(-s),

where x=1,…,N, s,b>0 are shape parameters, and N is the number of distinct 
categories. The Zipf distribution is just a special case of the 
Zipf-Mandelbrot distribution where the second shape parameter b=0. The zeta 
distribution has mass

p(x) = x^-λ/ζ(s),

where x=1,2,…, s>1 is the shape parameter, and ζ() is the Riemann zeta 
function given by:

ζ(t) = ∑_{i=1}^∞ 1/i^t<∞.

Note that the zeta distribution is just a special case of the Zipf 
distribution where s>1 and N goes to infinity.

Value
-----

    pzipfman gives the distribution function, for the specified distribution.

Note
----

    These functions may be updated in a future version of the package so as to 
    allow greater flexibility with the inputs.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Mandelbrot, B. B. (1965), Information Theory and Psycholinguistics. In B. 
        B. Wolman and E. Nagel, editors. Scientific Psychology, Basic Books.

    Young, D. S. (2013), Approximate Tolerance Limits for Zipf-Mandelbrot 
        Distributions, Physica A: Statistical Mechanics and its Applications, 
        392, 1702–1711.

    Zipf, G. K. (1949), Human Behavior and the Principle of Least Effort, 
        Hafner.

    Zornig, P. and Altmann, G. (1995), Unified Representation of Zipf 
        Distributions, Computational Statistics and Data Analysis, 19, 461–473.

Examples
## Randomly generated data from the Zipf distribution.

    pzipfman(q = 5, s = 2, N = 100)

## Randomly generated data from the Zipf-Mandelbrot distribution.

    pzipfman(q = [2,5,7], s = 2, b = 3, N = 100)

## Randomly generated data from the zeta distribution.

    pzipfman(q = [6,8], s = 1.3, N = np.inf)
    '''
    if N == None:
        return 'Must specify N.'
    q = np.array(q)
    q = np.floor(q)
    if length(q) == 1:
        q = np.array([q])
    temp = []#np.array([None,]*length(q))
    if b == None and N < np.inf:
        if s <= 0:
            return "Invalid value for s!"
        if any(q <= 0):
            temp[np.where(q<=0)] = 0
        if any(q > N):
            temp[np.where(q>N)] = 1
        if any(q > 0) and any(q <= N):
            ind = (np.where(q > 0) and np.where(q <= N))[0]
            for i in range(length(ind)):
                temp.append(dzipfman(x=range(1,int(q[ind[i]])+1), s = s, N = N))
    elif b != None:
        if s <= 0 or b < 0:
            return 'Invalid value for s and/or b!'
        if N == np.inf:
            return "N must be finite!"
        if any(q <= 0):
            temp[np.where(q <= 0)] = 0
        if any(q > N):
            temp[np.where(q > N)] = 1
        if any(q > 0) and any(q < N):
            ind = (np.where(q > 0) and np.where(q <= N))[0]
            for i in range(length(ind)):
                temp.append(dzipfman(x=range(1,int(q[ind[i]])+1), s = s, b = b, N = N))
    else:
        if s <= 1:
            return 'Invalid value for s!'
        if any(q <= 0):
            temp[np.where(q <= 0)] = 0
        if any(q == np.inf):
            temp[np.where(q == np.inf)] = 1
        if any(q > 0) and any(q < np.inf):
            ind = (np.where(q > 0) and np.where(q <= N))[0]
            for i in range(length(ind)):
                temp.append(dzipfman(x=range(1,int(q[ind[i]])+1), s = s, b = b, N = np.inf))
    if lowertail == False:
        for i in range(length(temp)):
            temp[i] = np.round(1-sum(np.round(temp[i],12)),8)
        if any(temp < 0):
            temp[np.where(temp<0)] = 0
        if any(temp > 1):
            temp[np.where(temp>1)] = 0
        if logp:
            temp = np.log(temp)
    elif logp:
        for i in range(length(temp)):
            temp[i] = np.round(np.log(temp[i][0]) + np.log(1+sum(temp[i][1:])/temp[i][0]),8)
    else:
        temp = np.array(list((map(sum,temp))))
        if any(temp) < 0:
            temp[np.where(temp<0)] = 0
        if any(temp) > 1:
            temp[np.where(temp>1)] = 0
    return temp

def qzipfman(p, s = 1, b = None, N = None, lowertail = True, logp = False):
    '''
Zipf-Mandelbrot Distributions

Description
    Quantile function for the Zipf, Zipf-Mandelbrot, and zeta distributions.

Usage
    qzipfman(p, s, b = None, N = None, lowertail = True, 
             logp = False)

    
Parameters
----------
    p: list
        Vector of probabilities.

    s, b: float
        The shape parameters, both of which must be greater than 0. b must be 
        specified for Zipf-Mandelbrot distributions.

    N: int
        The number of categories, which must be integer-valued for Zipf and 
        Zipf-Mandelbrot distributions. For a zeta distribution, N = Inf must 
        be used.

    logp: bool
        Logical vectors. If TRUE, then the probabilities are given as log(p).

    lowertail: bool
        Logical vector. If TRUE, then probabilities are P[X≤ x], else P[X>x].

Details
-------

The Zipf-Mandelbrot distribution has mass

    p(x) = (x + b)^-s/∑_{i=1}^{N}(i + b)^(-s),

where x=1,…,N, s,b>0 are shape parameters, and N is the number of distinct 
categories. The Zipf distribution is just a special case of the 
Zipf-Mandelbrot distribution where the second shape parameter b=0. The zeta 
distribution has mass

p(x) = x^-λ/ζ(s),

where x=1,2,…, s>1 is the shape parameter, and ζ() is the Riemann zeta 
function given by:

ζ(t) = ∑_{i=1}^∞ 1/i^t<∞.

Note that the zeta distribution is just a special case of the Zipf 
distribution where s>1 and N goes to infinity.

Value
-----
    qzipfman gives the quantile function

Note
----

    These functions may be updated in a future version of the package so as to 
    allow greater flexibility with the inputs.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Mandelbrot, B. B. (1965), Information Theory and Psycholinguistics. In B. 
        B. Wolman and E. Nagel, editors. Scientific Psychology, Basic Books.

    Young, D. S. (2013), Approximate Tolerance Limits for Zipf-Mandelbrot 
        Distributions, Physica A: Statistical Mechanics and its Applications, 
        392, 1702–1711.

    Zipf, G. K. (1949), Human Behavior and the Principle of Least Effort, 
        Hafner.

    Zornig, P. and Altmann, G. (1995), Unified Representation of Zipf 
        Distributions, Computational Statistics and Data Analysis, 19, 461–473.

Examples
## Randomly generated data from the Zipf distribution.

    qzipfman(p = 0.20, s = 2, N = 100, lowertail = False)
    
    qzipfman(p = 0.80, s = 2, N = 100)

## Randomly generated data from the Zipf-Mandelbrot distribution.

    qzipfman(p = 0.20, s = 2, b = 3, N = 100, lowertail = False)
    
    qzipfman(p = 0.80, s = 2, b = 3, N = 100)

## Randomly generated data from the zeta distribution.

    qzipfman(p = 0.20, s = 1.3, lowertail = False, N = np.inf)
    
    qzipfman(p = 0.80, s = 1.3, N = np.inf)
    '''
    if N == None:
        return 'Must specify N.'
    if N != np.inf:
        N = int(N)
    if length(p) == 1:
        p = [p]
    p = np.array(p)
    if logp:
        p = np.exp(p)
    if lowertail == False:
        p = 1-p
    if b == None and N < np.inf:
        if s <= 0:
            return "Invalid value for s!"
        allp = np.linspace(1,1,length(p))
        tempind = None
        if any(p > 1) or any(p < 0):
            tempind = np.where(p > 1) 
            allp[tempind] = np.nan
            tempind = np.where(p < 0)
            allp[tempind] = np.nan
        temp = ([pzipfman(q=1,s=s,N=N) >= a or a>1 for a in p])
        nottemp = [not t for t in temp]
        if sum(nottemp) > 0 and N < 1e6:
            ptemp = np.cumsum(dzipfman(x=np.array(range(1,N+1)), s = s, N = N))
            outp = []
            outp2 = []
            for i in range(temp.count(not True)):
                outp.append(min(np.where(ptemp >= p[np.where(nottemp)][i]))[0]+1) #not sure if +1 should be here, makes the same as R
                try:
                    outp2.append(max(np.where(ptemp == p[np.where(nottemp)][i]))[0]+1) ##
                except:
                    outp2.append(-np.inf)
            allp[np.where(nottemp)] = max(outp,outp2)
        elif sum(nottemp) > 0 and N >= 1e6:
            x = np.array([1e6,1e300])
            y = np.array([np.round(sum(dzipfman(x=np.array(range(1,int(1e6+1))),s=s,N=N)),8),1])
            whichp = np.where([a >= y[0] and a < 1 for a in p])[0]
            # #creating an lm object, 2 steps
            #  # 1.) make a dataframe (df)
            #  # 2.) lm_object: lm('y ~ x*', data = df) == ols('y ~ x*', data = df).fit()
            xtmp = (-1/x)
            df = pd.DataFrame({'xtmp':xtmp,'y':y})
            outlm = ols('y~xtmp',data=df).fit()
            be = outlm.params
            newq = -1/((p[whichp]-be[0])/be[1])
            if any(newq >= 1e300) or any(newq == np.inf):
                allp[np.where(newq >= 1e300)] = np.inf
            if any(newq < 1e300) and any(newq >= 1e6):
                allp[whichp[np.where(newq < 1e300) and np.where(newq >= 1e6)]] = np.floor(newq[np.where(newq < 1e300) and np.where(newq >= 1e6)])
                
    elif b != None:
        if s <= 0 or b < 0:
            return "Invalid value for s and/or b!"
        if N == np.inf:
            return "N must be finite"
        allp = np.linspace(1,1,length(p))
        tempind = None
        if any(p > 1) or any(p < 0):
            tempind = np.where(p > 1) 
            allp[tempind] = np.nan
            tempind = np.where(p < 0)
            allp[tempind] = np.nan
        temp = ([pzipfman(q=1,s=s,N=N) >= a or a>1 for a in p])
        nottemp = [not t for t in temp]
        if sum(nottemp)>0 and N < 1e6:
            ptemp = np.cumsum(dzipfman(x=np.array(range(1,N+1)), s = s, b = b, N = N))
            outp = []
            outp2 = []
            for i in range(temp.count(not True)):
                outp.append(min(np.where(ptemp >= p[np.where(nottemp)][i]))[0]+1) #not sure if +1 should be here, makes the same as R
                try:
                    outp2.append(max(np.where(ptemp == p[np.where(nottemp)][i]))[0]+1) ##
                except:
                    outp2.append(-np.inf)
            allp[np.where(nottemp)] = max(outp,outp2)
        elif sum(nottemp) > 0 and N >= 1e6:
            x = np.array([1e6,1e300])
            y = np.array([np.round(sum(dzipfman(x=np.array(range(1,int(1e6+1))),s=s,b=b,N=N)),8),1])
            whichp = np.where([a >= y[0] and a < 1 for a in p])[0]
            # #creating an lm object, 2 steps
            #  # 1.) make a dataframe (df)
            #  # 2.) lm_object: lm('y ~ x*', data = df) == ols('y ~ x*', data = df).fit()
            xtmp = (-1/x)
            df = pd.DataFrame({'xtmp':xtmp,'y':y})
            outlm = ols('y~xtmp',data=df).fit()
            be = outlm.params
            newq = -1/((p[whichp]-be[0])/be[1])
            if any(newq >= 1e300) or any(newq == np.inf):
                allp[np.where(newq >= 1e300)] = np.inf
            if any(newq < 1e300) and any(newq >= 1e6):
                allp[whichp[np.where(newq < 1e300) and np.where(newq >= 1e6)]] = np.floor(newq[np.where(newq < 1e300) and np.where(newq >= 1e6)])
    else:
        if s <= 1:
            return "Invalid value for s!"
        allp = np.array([None,]*length(p))
        if any(p>1) or any(p<0):
            tempind = np.where(p > 1) 
            allp[tempind] = np.nan
            tempind = np.where(p < 0)
            allp[tempind] = np.nan
        temp = (1/zeta(s)*np.cumsum(np.float_power(range(1,int(1e06+1)),-s)))
        temp1 = 1/zeta(s)
        tempmax = round(max(temp),7)
        if any(p <= temp1):
            allp[(np.where(p <= temp1) and np.where(p >= 0))][0] = 1
        if any(p == 1):
            allp[np.where(p == 1)] = np.inf
        if any(p > tempmax):
            x = np.array([1e6, 1e300])
            y = np.array([tempmax,1])
            xtmp = (-1/x)
            df = pd.DataFrame({'xtmp':xtmp,'y':y})
            outlm = ols('y~xtmp',data=df).fit()
            be = outlm.params
            whichp = np.where(p > tempmax) and np.where(p <= 1)
            newq = -1/((p[whichp]-be[0])/be[1])
            if any(newq >= 1e300) or any(newq == -np.inf):
                allp[np.where(newq >= 1e300)] = np.inf
            if any(newq < 1e300) and any(newq >= 1e6):
                allp[whichp[np.where(newq < 1e300) and np.where(newq >= 1e6)]] = np.floor(newq[np.where(newq < 1e300) and np.where(newq >= 1e6)])
        if any(allp == None):
            NoneArr = np.where(allp == None)[0]
            for i in range(list(allp).count(None)):
                allp[NoneArr[i]] = np.min((np.where(temp >= p[NoneArr[i]]))[0]+1)
        if any(allp == -np.inf):
            allp[np.where(allp = -np.inf)] = np.nan
    return allp

def rzipfman(n, s = 1, b = None, N = None):
    '''
Zipf-Mandelbrot Distributions

Description
    random generation for the Zipf, Zipf-Mandelbrot, and zeta distributions.

Usage
    rzipfman(n, s, b = None, N = None)
    
Parameters
----------
    n: int
        The number of observations. If length>1, then the length is taken to 
        be the number required.

    s, b: float
        The shape parameters, both of which must be greater than 0. b must be 
        specified for Zipf-Mandelbrot distributions.

    N: int
        The number of categories, which must be integer-valued for Zipf and 
        Zipf-Mandelbrot distributions. For a zeta distribution, N = Inf must 
        be used.
Details
-------

The Zipf-Mandelbrot distribution has mass

    p(x) = (x + b)^-s/∑_{i=1}^{N}(i + b)^(-s),

where x=1,…,N, s,b>0 are shape parameters, and N is the number of distinct 
categories. The Zipf distribution is just a special case of the 
Zipf-Mandelbrot distribution where the second shape parameter b=0. The zeta 
distribution has mass

p(x) = x^-λ/ζ(s),

where x=1,2,…, s>1 is the shape parameter, and ζ() is the Riemann zeta 
function given by:

ζ(t) = ∑_{i=1}^∞ 1/i^t<∞.

Note that the zeta distribution is just a special case of the Zipf 
distribution where s>1 and N goes to infinity.

Value
-----

    rzipfman generates random deviates for the specified distribution.

Note
----

    These functions may be updated in a future version of the package so as to 
    allow greater flexibility with the inputs.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Mandelbrot, B. B. (1965), Information Theory and Psycholinguistics. In B. 
        B. Wolman and E. Nagel, editors. Scientific Psychology, Basic Books.

    Young, D. S. (2013), Approximate Tolerance Limits for Zipf-Mandelbrot 
        Distributions, Physica A: Statistical Mechanics and its Applications, 
        392, 1702–1711.

    Zipf, G. K. (1949), Human Behavior and the Principle of Least Effort, 
        Hafner.

    Zornig, P. and Altmann, G. (1995), Unified Representation of Zipf 
        Distributions, Computational Statistics and Data Analysis, 19, 461–473.

Examples
## Randomly generated data from the Zipf distribution.

    rzipfman(n = 150, s = 2, N = 100)

## Randomly generated data from the Zipf-Mandelbrot distribution.

    rzipfman(n = 150, s = 2, b = 3, N = 100)

## Randomly generated data from the zeta distribution.

    rzipfman(n = 100, s = 1.3, N = np.inf)


    '''
    if N == None:
        return 'Must specify N.'
    if b == None and N < np.inf:
        if s <= 0:
            return "Invalid value for s!"
        out = qzipfman(p = st.uniform.rvs(size=n), s=s, N=N)
        out = [int(a) for a in out]
    elif b != None:
        if s <= 0 or b <0:
            return "Invalid value for s and/or b!"
        if N == np.inf:
            return "N must be finite!"
        out = qzipfman(p = st.uniform.rvs(size=n), s=s, b=b, N=N)
        out = [int(a) for a in out]
    else:
        if s <= 1:
            return "Invalid value for s!"
        out = qzipfman(p = st.uniform.rvs(size=n), s=s, N=np.inf)
        out = [int(a) for a in out]
    outlvl = pd.DataFrame(pd.Series(out).value_counts().sort_values(ascending=False)).T.columns
    y = list(range(length(np.unique(out))))
    out = np.array(out)
    indexes = []
    for i in range(length(outlvl)):
        indexes.append(np.where(out == outlvl[i]))
    for i in range(length(indexes)):
        out[indexes[i]] = i
    out = np.array([y[a]+1 for a in out])
    return out            
    

#print(rzipfman(n = 150, s = 2, N = 100))
#print(rzipfman(n = 150, s = 2, b = 3, N = 100))
#print(rzipfman(n = 100, s = 1.3, N = np.inf))
#print(qzipfman(.9,s=4.3,N=np.inf))
#print(qzipfman([.97,.98,.99,1.1,.999,0.95,1,-1,-2,0], N = np.inf,s=4.3))
#print(qzipfman(x=[1,2,3,0.5,0.7,1.25,101], N = np.inf,b=None,s=2))
    
