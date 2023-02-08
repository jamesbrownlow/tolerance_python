import numpy as np
from statsmodels.formula.api import ols
import pandas as pd
import scipy.optimize as opt
import scipy.stats as st
from scipy.special import zeta

def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    return len(x)

def zetafun(x):
    '''
Zeta Function

Description
    Zeta function, internal

Usage
    zetafun(x)
    
Parameters
----------
    x:
        For zetafun, a vector or matrix whose real values must be greater than 
        or equal to 1.

Details
-------
    This functions are not intended to be called by the user. zetafun is a 
    condensed version of the Riemann's zeta function given in R's VGAM package.
    Please use that reference if looking to directly implement Riemann's zeta 
    function. The function we have included is done so out of convenience.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.

    Yee, T. (2010), The VGAM Package for Categorical Data Analysis, Journal of 
        Statistical Software, 32, 1–34.
    
Example
-------
    zetafun([2,3,4,5,6])
    '''
    x = np.array(x)
    if any(x < 1): 
        return "Invalid input for Riemann's zeta function."
    a = 12
    k = 8
    B = [1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6, -3617/510]
    ans = []
    for i in range(1,a):
        ans.append(1/i**x[i%length(x)-1])
    ans = np.array(ans)
    ans = np.sum(ans)
    ans = ans + 1/((x - 1) * a**(x - 1)) + 1/(2 * a**x)
    term = (x/2)/a**(x + 1)
    ans = ans + term * B[0]
    for mm in range(1,k):
        term = term * (x + 2 * mm - 2) * (x + 2 * mm - 3)/(a * a * 2 * mm * (2 * mm - 1))
        ans = ans + term * B[mm]
    return ans

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
          

def zmll(x, N = None, s = 1, b = 1, dist = 'Zipf'):
    '''
Maximum Likelihood Estimation for Zipf-Mandelbrot Models

Description
    Performs maximum likelihood estimation for the parameters of the Zipf, 
    Zipf-Mandelbrot, and zeta distributions.

Usage
    zmll(x, N = None, s = 1, b = 1, dist = ["Zipf", "Zipf-Man", "Zeta"]) 
    
Parameters
----------
    x:	
        A vector of raw data or a table of counts which is distributed 
        according to a Zipf, Zipf-Mandelbrot, or zeta distribution. Do not 
        supply a vector of counts!

    N:
        The number of categories when dist = "Zipf" or dist = "Zipf-Man". This
        is not used when dist = "Zeta". If N = None, then N is estimated based
        on the number of categories observed in the data.

    s:
        The initial value to estimate the shape parameter, which is set to 1 
        by default. If a poor initial value is specified, then a warning 
        message is returned.

    b:	
        The initial value to estimate the second shape parameter when 
        dist = "Zipf-Man", which is set to 1 by default. If a poor initial 
        value is specified, then a warning message is returned.

    dist:
        Options are dist = "Zipf", dist = "Zipf-Man", or dist = "Zeta" if the 
        data is distributed according to the Zipf, Zipf-Mandelbrot, or zeta 
        distribution, respectively.

Details
    Zipf-Mandelbrot models are commonly used to model phenomena where the 
    frequencies of categorical data are approximately inversely proportional 
    to its rank in the frequency table.

Returns
-------
    zmll returns a dataframe with coefficients

Note
    This function may be updated in a future version of the package so as to 
    allow greater flexibility with the inputs.

References
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Mandelbrot, B. B. (1965), Information Theory and Psycholinguistics. In B. 
        B. Wolman and E. Nagel, editors. Scientific Psychology, Basic Books.

    Zipf, G. K. (1949), Human Behavior and the Principle of Least Effort, 
        Hafner.

    Zornig, P. and Altmann, G. (1995), Unified Representation of Zipf 
        Distributions, Computational Statistics and Data Analysis, 19, 461–473.

Examples
    ## Maximum likelihood estimation for randomly generated data
    ## from the Zipf, Zipf-Mandelbrot, and zeta distributions. 
        N = 30
        
        s = 2
        
        b = 5
        
        Zdata = [6, 2, 1, 4, 8, 3, 3, 14, 2, 1, 21, 5, 18, 2, 30, 10, 8, 2, 
                  11, 4, 16, 13, 17, 1, 7, 1, 1, 28, 19, 27, 2, 7, 7, 13, 1,
                  15, 1, 16, 9, 9, 7, 29, 3, 10, 3, 1, 20, 8, 12, 6, 11, 5, 1,
                  5, 23, 3, 3, 14, 6, 9, 1, 24, 5, 11, 15, 1, 5, 5, 4, 10, 1,
                  12, 1, 3, 4, 2, 9, 2, 1, 25, 6, 8, 2, 1, 1, 1, 4, 6, 7, 26, 
                  10, 2, 1, 2, 17, 4, 3, 22, 8, 2]
        
    ## Zipf
        zmll(x = Zdata, N = N, s = s, b = b, dist = 'Zipf')
    
    ## Zipf-Mandelbrot
        zmll(x = Zdata, N = N, s = s, b = b, dist = 'Zipf-Man')
    
    # Zeta
        zmll(x = Zdata, N = np.inf, s = s, b = b, dist = 'Zeta')
    '''
    x = pd.DataFrame(x)
    x = pd.DataFrame(x.value_counts()).T
    x.columns = list(range(0,length(x.iloc[0])))
    Ntemp = length(x.iloc[0])
    x = x.reindex(np.argsort(x.columns),axis=1)
    if dist == 'Zeta':
        N = Ntemp
    if N == None:
        N = Ntemp
    if N < Ntemp:
        return "N cannot be smaller than the maximun number of categories in x!"
    Nseq = np.array(list(range(1,N+1)))
    zeros = np.zeros(N-length(x.iloc[0]))
    zeros = [int(z) for z in zeros]
    zeros = pd.DataFrame(zeros).T
    zeros.columns = ['']*length(zeros.columns)
    x.iloc[0] = x.iloc[0]
    x = pd.concat([x,zeros],axis=1)
    x = x.iloc[0].to_numpy()
    if dist == 'Zipf':
        def llzipf(s):
            return sum(x*(s*np.log(Nseq)+np.log(sum(1/(Nseq)**s))))
        s = opt.minimize(llzipf, x0=0, method = 'BFGS')['x']
        vcov = opt.minimize(llzipf, x0=0, method = 'BFGS')['hess_inv'].ravel()
        fit = pd.DataFrame({'s':s,'vcov':vcov})
    if dist == "Zipf-Man":
        def llzima(params = [s,b]):
            return sum(x*(params[0]*np.log(Nseq+params[1])+np.log(sum(1/(Nseq+params[1])**params[0]))))
        s = opt.minimize(llzima, x0=[0,0],method = 'L-BFGS-B')['x']
        vcov = opt.minimize(llzima, x0=[0,0],method = 'L-BFGS-B')['hess_inv'].todense()
        fit = pd.DataFrame([s,vcov]).T
        fit.columns = ['Coefficients','vcov']
    if dist == "Zeta":
        def llzeta(s):
            return sum(x*(s*np.log(Nseq)+np.log(zetafun(s))))
        s = opt.minimize(llzeta, x0=1+1e-14, method = 'BFGS')['x']
        vcov = opt.minimize(llzeta, x0=1+1e-14, method = 'BFGS')['hess_inv'].ravel()
        fit = pd.DataFrame({'s':s,'vcov':vcov})
    return fit
 
    
def zipftolint(x, m = None, N = None, alpha = 0.05, P = 0.99, side = 1, s = 1, b = 1, dist = 'Zipf', ties = False):
    '''
Zipf-Mandelbrot Tolerance Intervals

Description
    Provides 1-sided or 2-sided tolerance intervals for data distributed 
    according to Zipf, Zipf-Mandelbrot, and zeta distributions.

Usage
    zipftolint(x, m = None, N = None, alpha = 0.05, P = 0.99, side = 1, s = 1, 
               b = 1, dist = ["Zipf", "Zipf-Man", "Zeta"], ties = False) 
    
Parameters
----------
    x: list
        A vector of raw data or a table of counts which is distributed 
        according to a Zipf, Zipf-Mandelbrot, or zeta distribution. Do not 
        supply a vector of counts!

    m: int, optional
        The number of observations in a future sample for which the tolerance limits will be calculated. By default, m = NULL and, thus, m will be set equal to the original sample size.

    N: int, optional
        The number of categories when dist = "Zipf" or dist = "Zipf-Man". This 
        is not used when dist = "Zeta". If N = None, then N is estimated based 
        on the number of categories observed in the data.

    alpha: float, optional
        The level chosen such that 1-alpha is the confidence level.

    P: float, optional
        The proportion of the population to be covered by this tolerance 
        interval.

    side: 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively).

    s: float, optional
        The initial value to estimate the shape parameter in the zmll function.

    b: float, optional
        The initial value to estimate the second shape parameter in the zmll 
        function when dist = "Zipf-Man".

    dist: string, optional	
        Options are dist = "Zipf", dist = "Zipf-Man", or dist = "Zeta" if the 
        data is distributed according to the Zipf, Zipf-Mandelbrot, or zeta 
        distribution, respectively.

    ties: bool, optional
        How to handle if there are other categories with the same frequency as 
        the category at the estimated tolerance limit. The default is False, 
        which does no correction. If TRUE, then the highest ranked 
        (i.e., lowest number) of the tied categories is selected for the lower 
        limit and the lowest ranked (i.e., highest number) of the tied 
        categories is selected for the upper limit.

Details
    Zipf-Mandelbrot models are commonly used to model phenomena where the 
    frequencies of categorical data are approximately inversely proportional 
    to its rank in the frequency table. Zipf-Mandelbrot distributions are 
    heavily right-skewed distributions with a (relatively) large mass placed 
    on the first category. For most practical applications, one will typically
    be interested in 1-sided upper bounds.

Returns
-------
    zipftolint returns a data frame with the following items:

        alpha
            The specified significance level.

        P	
            The proportion of the population covered by this tolerance 
            interval.

        shat	
            MLE for the shape parameter s.

        bhat	
            MLE for the shape parameter b when dist = "Zipf-Man".

        1-sided.lower	
            The 1-sided lower tolerance bound. This is given only if side = 1.
    
        1-sided.upper	
            The 1-sided upper tolerance bound. This is given only if side = 1.

        2-sided.lower	
            The 2-sided lower tolerance bound. This is given only if side = 2.

        2-sided.upper	
            The 2-sided upper tolerance bound. This is given only if side = 2.

Note
    This function may be updated in a future version of the package so as to 
    allow greater flexibility with the inputs.

References
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Mandelbrot, B. B. (1965), Information Theory and Psycholinguistics. 
        In B. B. Wolman and E. Nagel, editors. Scientific Psychology, 
        Basic Books.

    Young, D. S. (2013), Approximate Tolerance Limits for Zipf-Mandelbrot 
        Distributions, Physica A: Statistical Mechanics and its Applications, 
        392, 1702–1711.

    Zipf, G. K. (1949), Human Behavior and the Principle of Least Effort, 
        Hafner.

    Zornig, P. and Altmann, G. (1995), Unified Representation of Zipf 
        Distributions, Computational Statistics and Data Analysis, 19, 461–473.
        
Examples
    ## 95%/99% 1-sided tolerance intervals for the Zipf, Zipf-Mandelbrot, and 
    zeta distributions.
        N = 30  
        
        s = 2
        
        b = 5
        
        # Zipf
        
        Zdata = [6, 2, 1, 4, 8, 3, 3, 14, 2, 1, 21, 5, 18, 2, 30, 10, 8, 2, 
                 11, 4, 16, 13, 17, 1, 7, 1, 1, 28, 19, 27, 2, 7, 7, 13, 1,
                 15, 1, 16, 9, 9, 7, 29, 3, 10, 3, 1, 20, 8, 12, 6, 11, 5, 1,
                 5, 23, 3, 3, 14, 6, 9, 1, 24, 5, 11, 15, 1, 5, 5, 4, 10, 1,
                 12, 1, 3, 4, 2, 9, 2, 1, 25, 6, 8, 2, 1, 1, 1, 4, 6, 7, 26, 
                 10, 2, 1, 2, 17, 4, 3, 22, 8, 2] 
        
        zipftolint(x = Zdata, dist = 'Zipf', N = N, s=s, b=b)
        
        # Zipf-Mandelbrot
        
        Zdata = [2,2,2,2,2,2,2,2,3,3,2,2,2,2,2,3,2,3,2,3,2,3,2,2,3,3,3,4,5,6]
        
        zipftolint(x = Zdata, dist = 'Zipf-Man',side = 1)
        
        #Zeta
        
        Zdata = [0,1,4,3,1,1,1,1,1,2,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1]
        
        zipftolint(x = Zdata, dist = 'Zeta', N = N, s=s, b=b,side = 2)
    '''
    if side != 1 and side != 2:
        return 'Must specify a one-sided or two-sided procedure.'
    if side == 2:
        alpha = alpha/2
        P = (P+1)/2
    fit = zmll(x=x,N=N,s=s,b=b,dist=dist)
    x = pd.DataFrame(x)
    x = pd.DataFrame(x.value_counts()).T
    x.columns = list(range(0,length(x.iloc[0])))
    Ntemp = length(x.iloc[0])
    if N == None:
        N = Ntemp
    n = sum(x.iloc[0])
    if m == None:
        m = n
    if dist == 'Zipf':
        shat = fit.iloc[:,0][0]
        sse = np.sqrt(fit.iloc[:,1][0])
        TI1 = shat - st.norm.ppf(1-alpha)*sse*np.sqrt(n/m)
        TI2 = shat + st.norm.ppf(1-alpha)*sse*np.sqrt(n/m)
        lowers = max(TI1,0)
        uppers = TI2
        try:
            lower = np.max(qzipfman(1-P, s = uppers, N = N),1)
        except:
            lower = 1
        try:
            upper = np.min(qzipfman(P, s = lowers, N = N),N)
        except:
            upper = N
    if dist == 'Zeta':
        N = np.inf
        shat = fit.iloc[:,0][0]
        sse = np.sqrt(fit.iloc[:,1][0])
        TI1 = shat - st.norm.ppf(1-alpha)*sse*np.sqrt(n/m)
        TI2 = shat + st.norm.ppf(1-alpha)*sse*np.sqrt(n/m)
        lowers = max(TI1,0)
        uppers = TI2
        try:
            lower = np.max(qzipfman(1-P, s = uppers, N = np.inf),1)
        except:
            lower = 1
        try:
            upper = qzipfman(P, s = lowers, N = np.inf)
        except:
            upper = N
    if dist == 'Zipf-Man':
        shat = fit.iloc[:,0][0]
        sse = np.sqrt(fit.iloc[:,1][0][0])
        bhat = fit.iloc[:,0][1]
        bse = np.sqrt(fit.iloc[:,1][1][1])
        if bhat == 0:
            print("Warning: MLE for b is 0! Consider fitting a Zipf distribution")
        sCI1 = shat - st.norm.ppf(1-alpha)*sse*np.sqrt(n/m)
        sCI2 = shat + st.norm.ppf(1-alpha)*sse*np.sqrt(n/m)
        bCI1 = bhat - st.norm.ppf(1-alpha)*bse*np.sqrt(n/m)
        bCI2 = bhat + st.norm.ppf(1-alpha)*bse*np.sqrt(n/m)
        lowers = max(sCI1,1e-14)
        uppers = sCI2
        lowerb = max(bCI1,0)
        upperb = bCI2
        try:
            lower = np.max(qzipfman(1-P, s = uppers, b = lowerb, N = N),1)
        except:
            lower = 1
        try:
            upper = np.min(qzipfman(P, s = lowers, b = upperb, N = N), N)
        except:
            upper = N
        
    if side == 2:
        alpha = 2*alpha
        P = (2*P)-1
    if ties and length(x)>upper:
        upper = np.max(np.where(np.array(x) == x[upper]))
    if ties and lower > 1:
        lower = np.min(np.where(np.array(x) == x[lower]))
    if dist != 'Zipf-Man':
        temp = pd.DataFrame({"alpha":[alpha], "P":[P], "shat":shat, "1-sided.lower":lower,"1-sided.upper":upper})
        if side == 2:
            temp.columns = ["alpha", "P", "shat", "2-sided.lower","2-sided.upper"]
    else:
        temp = pd.DataFrame({"alpha":[alpha], "P":[P], "shat":shat, 'bhat':bhat, "1-sided.lower":lower,"1-sided.upper":upper})
        if side == 2:
            temp.columns = ["alpha", "P", "shat", "bhat", "2-sided.lower","2-sided.upper"]
    return temp
