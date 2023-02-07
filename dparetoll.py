import numpy as np
import scipy.stats
import pandas as pd
import scipy.interpolate as interpolate
import math
import scipy.optimize as opt
#need dpareto.R
def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    return len(x)

def ddpareto(x,theta,log=False):
    if theta <= 0 or theta >= 1:
        return('theta must be between 0 and 1')
    n = [np.where(y < 0) for y in x]    
    temp = []
    for i in range(len(n)):
        temp.extend(n[i][0])
    n = temp
    x1 = x
    x = [max(y,0) for y in x]
    x = np.array(x)
    p = theta**(np.log(1+x))-theta**(np.log(2+x))
    if len(n) > 0:
        p[n[0]] = 0
    if log:
        p = np.log(p)
    for i in range(len(p)):
        if x1[i] < 0:
            p[i] = 0
        else:
            p[i] = p[i]
    if not log:
        p = [min(max(y,0),1) for y in p]
    return np.array(p)

def dparetoll(x,theta=None):
    '''
Maximum Likelihood Estimation for the Discrete Pareto Distribution

Description
    Performs maximum likelihood estimation for the parameter of the discrete 
    Pareto distribution.

Usage
    dparetoll(x, theta = None) 
    
Parameters
----------
    x:
        A vector of raw data which is distributed according to a 
        Poisson-Lindley distribution.

    theta:
        Optional starting value for the parameter. If None, then the method of 
        moments estimator is used.

Details
    The discrete Pareto distribution is a discretized of the continuous Type 
    II Pareto distribution (also called the Lomax distribution).

Returns
-------
    dparetoll returns a dataframe with coefficient theta

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Krishna, H. and Pundir, P. S. (2009), Discrete Burr and Discrete Pareto 
        Distributions, Statistical Methodology, 6, 177–188.

    Young, D. S., Naghizadeh Qomi, M., and Kiapour, A. (2019), Approximate 
        Discrete Pareto Tolerance Limits for Characterizing Extremes in Count 
        Data, Statistica Neerlandica, 73, 4–21.

Examples
## Maximum likelihood estimation for randomly generated data
## from the discrete Pareto distribution. 
    dparetoll(x=[1,4,2,5,6,2,4,7,3,2])
    
    dparetoll(x=[1,4,2,5,6,2,4,7,3,2],theta = 0.2)
    '''
    if theta != None:
        if theta >= 1 or theta <= 0:
            return 'theta must be between 0 and 1. '
    thtable = pd.DataFrame({'0':[0,0.050,0.089,0.126,0.164,0.203,0.244,0.286,0.332,0.380,
                      0.431,0.486,0.545,0.609,0.678,0.753,0.835,0.925,1.025,1.135,
                      1.258,1.395,1.549,1.723,1.921,2.146,2.404,2.701,3.044,3.442,
                      3.904,4.442,5.071,5.807,6.669,7.682,8.870],'1':np.arange(0,.3601,0.01)})
    x = np.array(x)
    inftest = np.where(x==np.inf)[0]
    if len(inftest)>0:
        noinf = np.where(x != np.inf)[0]
        for i in range(len(inftest)):
            x[inftest[i]] = np.max(x[noinf])
        print("Values of x equal to 'Inf' are set to the maximum finite value.")
    xbar = np.mean(x)
    if theta == None:
        if xbar <= max(thtable['0']):
            ind = np.max(np.where(thtable['0']<xbar))
            y= interpolate.interp1d(np.array(thtable.iloc[ind:ind+2]['0']), np.array(thtable.iloc[ind:ind+2]['1']))
            theta = y(xbar)
        else:
            Shat = np.array(pd.DataFrame([np.mean(x>=x[i]) for i in range(len(x))]))
            try:
                theta = math.prod([y**((sum(np.log(1+x)))**(-1)) for y in Shat])
            except TypeError:
                return 'an x[i] in the inputted list is too large, do not exceed 17,999,999,999,999,999,999'
    tmp = np.where(ddpareto(x, theta=theta,log=True)==-np.inf)
    tmp = tmp[0]
    xtmp = np.where(ddpareto(x, theta=theta,log=True)!=-np.inf)
    maxx = max(x[xtmp[0]])
    if length(tmp) > 0:
        x[tmp] = maxx
        print("Numerical overflow problem when calculating log-density of some x values.  The problematic values are set to the maximum finite value calculated.")
    def llf(theta):
        return -sum([sum((ddpareto(x,theta,log=True)))])
    fit = opt.minimize(llf, x0 = theta, method = 'Nelder-Mead')['x']
    fit = pd.DataFrame(fit)
    fit.index = ['Coefficient']
    fit.columns = ['theta']
    return fit

#print(ddpareto(x=[-1,2,3,0,-12,12,45,3,1,2,43,53,23,5,35,3,43,4,64,53,4,34,243,2,32,42,5,46,45,32,4,42,44,24,21,55,-99],theta = 0.7))
#dparetoll(x=[1,np.inf,2,3,4,5,999,np.inf,239,24,65,12,4,53,64,7,68,6,85,47,435,3,43,5,46,45,7,48,549,59,67,97,np.inf,23,465,46,6,5,75,86])
#print(dparetoll(x=[1,3,4,5,999,239,24,65,12,4,53,64,7,68,6,85,47,435,3,43,5,46,45,7,48,549,59,67,97,23,465,46,6,5,75,86]))
#print(dparetoll(x=[1,4,2,5,6,2,4,7,3,2]))
