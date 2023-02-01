import numpy as np
import scipy.stats as st


def length(x=None):
    try:
        return len(x)
    except:
        if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
            return 1
        else:
            return 0       

def dpoislind(x, theta, log = False):
    '''
Discrete Poisson-Lindley Distribution

Description
    Density (mass) for the Poisson-Lindley distribution.

Usage
    dpoislind(x, theta, log = False)
    
Parameters
----------

    x: list	
        Vector of quantiles.

    theta: float	
        The shape parameter, which must be greater than 0.

    log: bool
        Logical vectors. If True, then the probabilities are given as log(p).

Details
    The Poisson-Lindley distribution has mass

        p(x) = (θ^2(x + θ + 2))/(θ + 1)^(x+3),

    where x=0,1,… and θ>0 is the shape parameter.

Returns
-------
    dpoislind gives the density (mass) for the specified distribution.

References
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Ghitany, M. E. and Al-Mutairi, D. K. (2009), Estimation Methods for the 
        Discrete Poisson-Lindley Distribution, Journal of Statistical 
        Computation and Simulation, 79, 1–9.

    Sankaran, M. (1970), The Discrete Poisson-Lindley Distribution, Biometrics,
        26, 145–149.
Examples
--------
    dpoislind(x=[-3,-2,-2,1,-1,0,1,1,1,2,2,3,1], theta = 0.5)
    
    '''
    if theta <= 0:
        return "theta must be positive!"
    if length(x) == 1:
        x = [x]
    x = np.array(x)
    p = (theta**2*(x+theta+2)/(theta+1)**(x+3))*(x>=0)
    if log:
        p = log(p)
    if not log:
        p = np.minimum(np.maximum(p,0),1)
    return p

def ppoislind(q, theta, lowertail = True, logp = False):
    '''
Discrete Poisson-Lindley Distribution

Description
    Distribution function for the Poisson-Lindley distribution.

Usage
    ppoislind(q, theta, lowertail = True, logp = False)
    
Parameters
----------

    q: list
        Vector of quantiles.

    theta: float	
        The shape parameter, which must be greater than 0.

    logp: bool
        Logical vectors. If True, then the probabilities are given as log(p).

    lowertail: bool	
        Logical vector. If True, then probabilities are P[X≤ x], else P[X>x].

Details
    The Poisson-Lindley distribution has mass

        p(x) = (θ^2(x + θ + 2))/(θ + 1)^(x+3),

    where x=0,1,… and θ>0 is the shape parameter.

Returns
-------
    ppoislind gives the distribution function for the specified distribution.

References
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Ghitany, M. E. and Al-Mutairi, D. K. (2009), Estimation Methods for the 
        Discrete Poisson-Lindley Distribution, Journal of Statistical 
        Computation and Simulation, 79, 1–9.

    Sankaran, M. (1970), The Discrete Poisson-Lindley Distribution, Biometrics,
        26, 145–149.
Examples
--------
    ppoislind(q = [-3,-2,-2,1,-1,0,1,1,1,2,2,3,1,-1], theta = 0.5,
              lowertail=False)

    '''
    if theta <= 0:
        return "theta must be positive!"
    if length(q) == 1:
        q = [q]
    q = np.array(q)
    ind = q<0
    q = [int(np.floor(a)) for a in q]
    temp = []
    for i in range(length(q)):
        temp.append(np.sum(dpoislind(x = range(q[i]+1),theta=theta,log=False)))
    if length(temp) == 1:
        temp = [temp]
    temp = np.array(temp)
    #temp[np.where(temp == 0)] = np.min(temp[np.where(temp != 0)])
    if lowertail == False:
        temp = 1-temp
    if any(ind):
        temp[ind] = 0 + 1 *(not lowertail)
    if logp:
        temp = np.log(temp)
    if not logp:
        temp = np.minimum(np.maximum(temp,0),1)
    return temp

def qpoislind(p, theta, lowertail = True, logp = False):
    '''
    Discrete Poisson-Lindley Distribution

Description
    Quantile function, and random for the Poisson-Lindley distribution.

Usage
    qpoislind(p, theta, lowertail = True, logp = False)
    
Parameters
----------
    p: list
        Vector of probabilities.

    theta: float
        The shape parameter, which must be greater than 0.

    logp: bool
        Logical vectors. If True, then the probabilities are given as log(p).

    lowertail: bool	
        Logical vector. If True, then probabilities are P[X≤ x], else P[X>x].

Details
    The Poisson-Lindley distribution has mass

        p(x) = (θ^2(x + θ + 2))/(θ + 1)^(x+3),

    where x=0,1,… and θ>0 is the shape parameter.

Returns
-------
    qpoislind gives the quantile function for the specified distribution.

References
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Ghitany, M. E. and Al-Mutairi, D. K. (2009), Estimation Methods for the 
        Discrete Poisson-Lindley Distribution, Journal of Statistical 
        Computation and Simulation, 79, 1–9.

    Sankaran, M. (1970), The Discrete Poisson-Lindley Distribution, Biometrics,
        26, 145–149.
Examples
--------
    qpoislind(p = .2,theta = 0.5,lowertail=(False))
    qpoislind(p = [0.80,.2,1,0,1.1,-1,-2,.5,.9999,.9,0,1,-1,0,.32,1,3], theta = 0.5,lowertail = True)
    '''
    if theta <= 0:
        return "theta must be positive!"
    if length(p) == 1:
        p = [p]
    p = np.array(p)
    if logp:
        p = np.exp(p)
    if theta > 0.125:
        up = 400
    else:
        up = 2000
    if lowertail:
        tmp = ppoislind(range(up+1),theta=theta)
        allp = []
        for i in range(length(p)):
            allp.append(min(np.where(tmp>=p[i])))
        allp = np.array([min(a) if length(a) > 0 else np.nan for a in allp])
        if length(p) > 1:
            allp[np.where(p == 1)] = np.inf
            allp[np.where(p == 0)] = 0 
            allp[np.where(p > 1)] = np.nan
            allp[np.where(p < 0)] = np.nan
        else:
            if(p == 1):
                allp = [np.inf]
            elif(p == 0):
                allp = [0]
            elif(p>1 or p<0):
                allp = [np.nan]
    else:
        tmp = ppoislind(range(up+1),theta=theta,lowertail = False)
        allp = []
        for i in range(length(p)):
            allp.append(np.maximum(max(np.where(tmp>p[i])),0)+1)
        allp = np.array([max(a) if length(a) > 0 else 0.0 for a in allp])
        if length(p) > 1:
            if(up ==2000) and any(allp == 2000):
                allp[np.where(allp==2000)] = np.inf
            allp[np.where(p == 1)] = 0
            allp[np.where(p == 0)] = np.inf
            allp[np.where(p > 1)] = np.nan
            allp[np.where(p < 0)] = np.nan
        else:
            if(up ==2000) and allp == 2000:
                allp = np.inf
            if(p == 1):
                allp = [0]
            elif(p == 0):
                allp = [np.inf]
            elif(p > 1 or p < 0):
                allp = [np.nan]
    if any(np.isnan(allp)):
        print("Warning message:\n NaN(s) produced")
    return allp
        
def rpoislind(n, theta):
    '''
Discrete Poisson-Lindley Distribution

Description
    Random generation for the Poisson-Lindley distribution.

Usage
    rpoislind(n, theta)
    
Parameters
----------
    n: int
        The number of observations. If length>1, then the sum of n\'s is used
        in place of n.

    theta: float
        The shape parameter, which must be greater than 0.

Details
    The Poisson-Lindley distribution has mass

        p(x) = (θ^2(x + θ + 2))/(θ + 1)^(x+3),

    where x=0,1,… and θ>0 is the shape parameter.

Returns
-------
    rpoislind generates random deviates for the specified distribution.

References
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Ghitany, M. E. and Al-Mutairi, D. K. (2009), Estimation Methods for the 
        Discrete Poisson-Lindley Distribution, Journal of Statistical 
        Computation and Simulation, 79, 1–9.

    Sankaran, M. (1970), The Discrete Poisson-Lindley Distribution, Biometrics,
        26, 145–149.
Examples
--------
    rpoislind(n = 150, theta = 0.5)
    rpoislind(n = [4,6], theta = 0.5)

    '''
    if theta <= 0:
        return "theta must be positive!"
    if length(n) > 1:
        n = sum(n)
    u = st.uniform.rvs(size = n)
    p = theta/(theta+1)
    ind = u > p 
    lamb = st.expon.rvs(theta, size = n) + (st.expon.rvs(theta, size = n))*ind
    out = st.poisson.rvs(lamb, size = n)
    return out

#print(rpoislind(n = 6, theta = 0.5))

#print(qpoislind(p = .2,theta = 0.5,lowertail=(False)))
#print(qpoislind(p = [0.80,.2,1,0,1.1,-1,-2,.5,.9999,.9,0,1,-1,0,.32,1,3], theta = 0.5,lowertail = True))

#print(ppoislind(q = [-3,-2,-2,1,-1,0,1,1,1,2,2,3,1,-1], theta = 0.5,lowertail=False))
#print(dpoislind(x=[-3,-2,-2,1,-1,0,1,1,1,2,2,3,1], theta = 0.5))