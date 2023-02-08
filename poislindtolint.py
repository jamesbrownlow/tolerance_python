import numpy as np
import scipy.stats as st
import scipy.optimize as opt
import pandas as pd

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

def poislindll(x, theta = None):
    '''
Maximum Likelihood Estimation for the Discrete Poisson-Lindley Distribution

Description
    Performs maximum likelihood estimation for the parameter of the 
    Poisson-Lindley distribution.

Usage
    poislindll(x, theta = None)
    
Parameters
----------
    x: list
        A vector of raw data which is distributed according to a 
        Poisson-Lindley distribution.

    theta: float, optional
        Optional starting value for the parameter. If None, then the method of 
        moments estimator is used.

Details
    The discrete Poisson-Lindley distribution is a compound distribution that,
    potentially, provides a better fit for count data relative to the
    traditional Poisson and negative binomial distributions.

Returns
-------
    poislindll returns a dataframe with items:
        s:
            The estimated coefficient value of theta. 
        vcov: 
            The variance-covariance matrix of the main parameters of a fitted 
            model object. In this case, it's the inverse of the hessian matrix.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Ghitany, M. E. and Al-Mutairi, D. K. (2009), Estimation Methods for 
        the Discrete Poisson-Lindley Distribution, Journal of Statistical 
        Computation and Simulation, 79, 1–9.

    Sankaran, M. (1970), The Discrete Poisson-Lindley Distribution, 
        Biometrics, 26, 145–149.

Examples
--------
    ## Maximum likelihood estimation for randomly generated data from the 
    Poisson-Lindley distribution. 
        Pdata = [2,1,3,2,0,11,9,0,0,0]
        
        poislindll(x = Pdata)
    '''
    x = pd.DataFrame(x)
    x = pd.DataFrame(x.value_counts()).T
    x = x.reindex(sorted(x.columns), axis=1)
    N = list(x.iloc[0])
    x = [list(x) for x in x.columns.values]
    x = [x[0] for x in x]
    N = np.array(N)
    x = np.array(x)
    Ntimesx = []
    for i in range(length(N)):
        Ntimesx.append(N[i]*x[i])
    xbar = sum(Ntimesx)/sum(N)
    if theta == None:
        theta = (-(xbar-1)+np.sqrt((xbar-1)**2+8*xbar))/(2*xbar)
    def llf(theta):
        return -2*sum(N)*np.log(theta)-sum(N*(np.log(x+theta+2)-np.log(theta+1)*(x+3)))
    try:
        s = opt.minimize(llf, x0=theta, method = 'BFGS')['x']
        vcov = opt.minimize(llf, x0=theta, method = 'BFGS')['hess_inv'].ravel()
        fit = pd.DataFrame({'s':s,'vcov':vcov})
    except:
        try:
            s = opt.minimize(llf, x0=theta*0.8, method = 'BFGS')['x']
            vcov = opt.minimize(llf, x0=theta*0.8, method = 'BFGS')['hess_inv'].ravel()
            fit = pd.DataFrame({'s':s,'vcov':vcov})
        except:
            return "Difficulty optimizing the MLE -- must try a different starting value for theta."
    return fit

def poislindtolint(x, m = None, alpha = 0.05, P = 0.99, side = 1):
    '''
    Poisson-Lindley Tolerance Intervals
    
Description
    Provides 1-sided or 2-sided tolerance intervals for data distributed 
    according to the Poisson-Lindley distribution.

Usage
    poislindtolint(x, m = None, alpha = 0.05, P = 0.99, side = 1)
Parameters
----------
    x: list
        A vector of raw data which is distributed according to a 
        Poisson-Lindley distribution.

    m: int, optional
        The number of observations in a future sample for which the tolerance 
        limits will be calculated. By default, m = NULL and, thus, m will be 
        set equal to the original sample size.

    alpha: float optional
        The level chosen such that 1-alpha is the confidence level.

    P: float, optional
        The proportion of the population to be covered by this tolerance 
        interval.

    side: 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively).

Details
    The discrete Poisson-Lindley distribution is a compound distribution that,
    potentially, provides a better fit for count data relative to the 
    traditional Poisson and negative binomial distributions. Poisson-Lindley 
    distributions are heavily right-skewed distributions. For most practical 
    applications, one will typically be interested in 1-sided upper bounds.

Returns
-------
    poislindtolint returns a data frame with the following items:
        alpha	
            The specified significance level.

        P	
            The proportion of the population covered by this tolerance 
            interval.

        theta	
            MLE for the shape parameter theta.

        1-sided.lower	
            The 1-sided lower tolerance bound. This is given only if side = 1.

        1-sided.upper	
            The 1-sided upper tolerance bound. This is given only if side = 1.

        2-sided.lower	
            The 2-sided lower tolerance bound. This is given only if side = 2.

        2-sided.upper	
            The 2-sided upper tolerance bound. This is given only if side = 2.

References
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Naghizadeh Qomi, M., Kiapour, A., and Young, D. S. (2015), Approximate 
        Tolerance Intervals for the Discrete Poisson-Lindley Distribution, 
        Journal of Statistical Computation and Simulation, 86, 841–854.
Examples
    ## 90%/90% 1-sided tolerance intervals for data assuming the 
    Poisson-Lindley distribution.
        #Pdata = [2,1,3,2,0,11,9,0,0,0]    

        Pdata = list([list(np.linspace(0,0,447)), list(np.linspace(1,1,132)), list(np.linspace(2,2,42)), list(np.linspace(3,3,21)), [4,4,4,5,5]])
        
        Pdata = [int(item) for sublist in Pdata for item in sublist]
        
        poislindtolint(x = Pdata, side = 2, m = 100, alpha = .10, P = 0.90)
    '''
    if side != 1 and side != 2:
        return "Must specify a one-sided or two-sided procedure!"
    n = length(x)
    if (side == 2):
        alpha = alpha/2
        P = (P + 1)/2
    if m == None:
        m = n
    out = poislindll(x)
    theta = out.iloc[:,0][0]
    vcov = out.iloc[:,1][0]
    TI1 = np.max(theta-st.norm.ppf(1-alpha)*np.sqrt(vcov)*np.sqrt(n/m),0)
    TI2 = np.max(theta+st.norm.ppf(1-alpha)*np.sqrt(vcov)*np.sqrt(n/m),0)
    lower = max(qpoislind(1-P, TI2),0)
    upper = qpoislind(P, TI1)
    if side == 2:
        alpha = 2*alpha
        P = (2*P) - 1
    temp = pd.DataFrame({"alpha":[alpha], "P":[P], "theta":theta, "2-sided.lower":lower, "2-sided.upper":upper})
    if side == 1:
        temp.columns = ["alpha", "P", "theta", "1-sided.lower", "1-sided.upper"]
    return temp
        
    
