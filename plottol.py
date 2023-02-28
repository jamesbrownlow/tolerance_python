#%matplotlib qt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.stats 
import scipy.integrate as integrate
import scipy.optimize as opt
import warnings
import statistics
warnings.filterwarnings('ignore')
import warnings
from scipy.optimize import curve_fit
import math


warnings.filterwarnings('ignore')

def Kfactor(n, f = None, alpha = 0.05, P = 0.99, side = 1, method = 'HE', m=50):
    K=None
    if f == None:
        f = n-1
    if (len((n,)*1)) != len((f,)*1) and (len((f,)*1) > 1):
        return 'Length of \'f\' needs to match length of \'n\'!'
    if (side != 1) and (side != 2):
        return 'Must specify one sided or two sided procedure'
    if side ==1:
        zp = scipy.stats.norm.ppf(P)
        ncp = np.sqrt(n)*zp
        ta = scipy.stats.nct.ppf(1-alpha,df = f, nc=ncp) #students t noncentralized
        K = ta/np.sqrt(n)
    else:
        def Ktemp(n, f, alpha, P, method, m):
            chia = scipy.stats.chi2.ppf(alpha, df = f)
            k2 = np.sqrt(f*scipy.stats.ncx2.ppf(P,df=1,nc=(1/n))/chia) #noncentralized chi 2 (ncx2))
            if method == 'HE':
                def TEMP4(n, f, P, alpha):
                    chia =  scipy.stats.chi2.ppf(alpha, df = f)
                    zp = scipy.stats.norm.ppf((1+P)/2)
                    za = scipy.stats.norm.ppf((2-alpha)/2)
                    dfcut = n**2*(1+(1/za**2))
                    V = 1 + (za**2)/n + ((3-zp**2)*za**4)/(6*n**2)
                    K1 = (zp * np.sqrt(V * (1 + (n * V/(2 * f)) * (1 + 1/za**2))))
                    G = (f-2-chia)/(2*(n+1)**2)
                    K2 = (zp * np.sqrt(((f * (1 + 1/n))/(chia)) * (1 + G)))
                    if f > dfcut:
                        K = K1
                    else:
                        K = K2
                        if K == np.nan or K == None:
                            K = 0
                    return K
                #TEMP5 = np.vectorize(TEMP4())
                K = TEMP4(n, f, P, alpha)
                return K
                
            elif method == 'HE2':
                zp = scipy.stats.norm.ppf((1+P)/2)
                K = zp * np.sqrt((1+1/n)*f/chia)
                return K
            
            elif method == 'WBE':
                r = 0.5
                delta = 1
                while abs(delta) > 0.00000001:
                    Pnew = scipy.stats.norm.cdf(1/np.sqrt(n)+r) - scipy.stats.norm.cdf(1/np.sqrt(n)-r)
                    delta = Pnew-P
                    diff = scipy.stats.norm.pdf(1/np.sqrt(n)+r) + scipy.stats.norm.pdf(1/np.sqrt(n)-r)
                    r = r-delta/diff
                K = r*np.sqrt(f/chia)
                return K
            
            elif method == 'ELL':
                if f < n**2:
                    print("Warning Message:\nThe ellison method should only be used for f appreciably larger than n^2")
                r = 0.5
                delta = 1
                zp = scipy.stats.norm.ppf((1+P)/2)
                while abs(delta) > 0.00000001:
                    Pnew = scipy.stats.norm.cdf(zp/np.sqrt(n)+r) - scipy.stats.norm.cdf(zp/np.sqrt(n)-r)
                    delta = Pnew - P
                    diff =  scipy.stats.norm.pdf(zp/np.sqrt(n)+r) +  scipy.stats.norm.pdf(zp/np.sqrt(n)-r)
                    r = r-delta/diff
                K = r*np.sqrt(f/chia)
                return K
            elif method == 'KM':
                K = k2
                return K
            elif method == 'OCT':
                delta = np.sqrt(n)*scipy.stats.norm.ppf((1+P)/2)
                def Fun1(z,P,ke,n,f1,delta):
                    return (2 * scipy.stats.norm.cdf(-delta + (ke * np.sqrt(n * z))/(np.sqrt(f1))) - 1) * scipy.stats.chi2.pdf(z,f1) 
                def Fun2(ke, P, n, f1, alpha, m, delta):
                    if n < 75:
                        return integrate.quad(Fun1,a = f1 * delta**2/(ke**2 * n), b = np.inf, args=(P,ke,n,f1,delta),limit = m)
                    else:
                        return integrate.quad(Fun1,a = f1 * delta**2/(ke**2 * n), b = n*1000, args=(P,ke,n,f1,delta),limit = m)
                def Fun3(ke,P,n,f1,alpha,m,delta):
                    f = Fun2(ke = ke, P = P, n = n, f1 = f1, alpha = alpha, m = m, delta = delta)
                    return abs(f[0] - (1-alpha))
                K = opt.minimize(fun=Fun3, x0=k2,args=(P,n,f,alpha,m,delta), method = 'L-BFGS-B')['x']
                return float(K)
            elif method == 'EXACT':
                def fun1(z,df1,P,X,n):
                    k = (scipy.stats.chi2.sf(df1*scipy.stats.ncx2.ppf(P,1,z**2)/X**2,df=df1)*np.exp(-0.5*n*z**2))
                    return k
                def fun2(X,df1,P,n,alpha,m):
                    return integrate.quad(fun1,a =0, b = 5, args=(df1,P,X,n),limit=m)
                def fun3(X,df1,P,n,alpha,m):
                    return np.sqrt(2*n/np.pi)*fun2(X,df1,P,n,alpha,m)[0]-(1-alpha)
                K = opt.brentq(f=fun3,a=0,b=k2+(1000)/n, args=(f,P,n,alpha,m))
                return K
        K = Ktemp(n=n,f=f,alpha=alpha,P=P,method=method,m=m)
    return K

def normtolint(x, alpha = 0.05, P = 0.99, side = 1, method = 'HE', m = 50, lognorm = False):
    '''
    normtolint(x, alpha = 0.05, P = 0.99, side = 1, method = ["HE", "HE2", "WBE", "ELL", "KM", "EXACT", "OCT"], m = 50, lognorm = False):
        
Parameters
----------
    x: list
        A vector of data which is distributed according to either a normal 
        distribution or a log-normal distribution.
    
    alpha: float, optional
        The level chosen such that 1-alpha is the confidence level.
        The default is 0.05.
    
    P: float, optional
        The proportion of the population to be covered by this tolerance 
        interval. The default is 0.99.
    
    side: 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively). The default is 1.
    
    method: string, optional
        The method for calculating the k-factors. The k-factor for the 1-sided 
        tolerance intervals is performed exactly and thus is the same for the 
        chosen method. 
        
            "HE" is the Howe method and is often viewed as being extremely 
            accurate, even for small sample sizes. 
        
            "HE2" is a second method due to Howe, which performs similarly to the 
            Weissberg-Beatty method, but is computationally simpler. 
        
            "WBE" is the Weissberg-Beatty method 
            (also called the Wald-Wolfowitz method), which performs similarly to 
            the first Howe method for larger sample sizes. 
            
            "ELL" is the Ellison correction to the Weissberg-Beatty method when f 
            is appreciably larger than n^2. A warning message is displayed if f is
            not larger than n^2. "KM" is the Krishnamoorthy-Mathew approximation 
            to the exact solution, which works well for larger sample sizes. 
            
            "EXACT" computes the k-factor exactly by finding the integral solution 
            to the problem via the integrate function. Note the computation time 
            of this method is largely determined by m. 
            
            "OCT" is the Owen approach to compute the k-factor when controlling 
            the tails so that there is not more than (1-P)/2 of the data in each 
            tail of the distribution.
            
        The default is "HE"
    
    m: int, optional 
        The maximum number of subintervals to be used in the integrate 
        function. This is necessary only for method = "EXACT" and method = 
        "OCT". The larger the number, the more accurate the solution. Too low 
        of a value can result in an error. A large value can also cause the 
        function to be slow for method = "EXACT". The default is m = 50.

    lower: float, optional
        If TRUE, then the data is considered to be from a log-normal 
        distribution, in which case the output gives tolerance intervals for 
        the log-normal distribution. The default is False.
    
Details
    Recall that if the random variable X is distributed according to a 
    log-normal distribution, then the random variable Y = ln(X) is distributed 
    according to a normal distribution.
    
Returns
-------
  normtolint returns a data frame with items:
        
    alpha: 
        The specified significance level.
    P: 
        The proportion of the population covered by this tolerance interval.
    mean:
        The sample mean.
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
    
    Howe, W. G. (1969), Two-Sided Tolerance Limits for Normal Populations - 
        Some Improvements, Journal of the American Statistical Association, 
        64, 610–620.

    Wald, A. and Wolfowitz, J. (1946), Tolerance Limits for a Normal 
        Distribution, Annals of Mathematical Statistics, 17, 208–215.

    Weissberg, A. and Beatty, G. (1969), Tables of Tolerance Limit Factors 
        for Normal Distributions, Technometrics, 2, 483–500.
        
Examples
--------
    ## 95%/95% 2-sided normal tolerance intervals for a sample of size 100. 
    
        x = np.random.normal(size=100)
        
        normtolint(x, alpha = 0.05, P = 0.95, side = 2, 
                    method = "HE", log.norm = FALSE)
    '''
    if lognorm:
        x = np.log(x)
    xbar = np.mean(x)
    s = statistics.stdev(x)
    n = len(x)
    K = Kfactor(n, alpha=alpha, P=P, side = side, method= method, m = m)
    lower = xbar-s*K
    upper = xbar+s*K
    if(lognorm):
        lower = np.exp(lower)
        upper = np.exp(upper)
        xbar = np.exp(xbar)
    if side == 1:
        temp = pd.DataFrame([[alpha,P, xbar,lower,upper]],columns=['alpha','P','mean','1-sided.lower','1-sided.upper'])
        return temp
    else:
        temp = pd.DataFrame([[alpha,P, xbar,lower,upper]],columns=['alpha','P','mean','2-sided.lower','2-sided.upper'])
        return temp

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import scipy.stats as stats

def levels(data,out):
    rownames = pd.Series(out.index)
    levels = []
    st = []
    for i in range(1,len(rownames)):
        levels.append(np.unique(np.array(data.iloc[:,i])))
        st.append([rownames[i-1],[levels[i-1]]])
    return st        
    
def to_int(x):
    try:
        return int(x)
    except:
        return 0    

def anovatolint(lmout, data, alpha = 0.05, P = 0.99, side = 1, method = 'HE', m = 50):
    '''
Tolerance Intervals for ANOVA

Description
    Tolerance intervals for each factor level in a balanced 
    (or nearly-balanced) ANOVA.
    
Usage
    anovatolint(lmout, data, alpha = 0.05, P = 0.99, side = 1,
                method = ["HE", "HE2", "WBE", "ELL", "KM", "EXACT", "OCT"], 
                m = 50)

    

Parameters
----------
    lmout : lm object - ols('y~x*',data=data).fit()
        An object of class lm (i.e., the results from the linear model fitting 
        routine such that the anova function can act upon).
        
    data : dataframe
        A data frame consisting of the data fitted in lm.out. Note that data 
        must have one column for each main effect (i.e., factor) that is 
        analyzed in lmout and that these columns must be of class factor.
        
    alpha : float, optional
        The level chosen such that 1-alpha is the confidence level. The 
        default is 0.05.
        
    P : float, optional
        The proportion of the population to be covered by this tolerance 
        interval. The default is 0.99.
        
    side : TYPE, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively). The default is 1.
        
    method : string, optional
        The method for calculating the k-factors. The k-factor for the 1-sided 
        tolerance intervals is performed exactly and thus is the same for the 
        chosen method. 
        
            "HE" is the Howe method and is often viewed as being extremely 
            accurate, even for small sample sizes. 
        
            "HE2" is a second method due to Howe, which performs similarly to the 
            Weissberg-Beatty method, but is computationally simpler. 
        
            "WBE" is the Weissberg-Beatty method 
            (also called the Wald-Wolfowitz method), which performs similarly to 
            the first Howe method for larger sample sizes. 
            
            "ELL" is the Ellison correction to the Weissberg-Beatty method when f 
            is appreciably larger than n^2. A warning message is displayed if f is
            not larger than n^2. "KM" is the Krishnamoorthy-Mathew approximation 
            to the exact solution, which works well for larger sample sizes. 
            
            "EXACT" computes the k-factor exactly by finding the integral solution 
            to the problem via the integrate function. Note the computation time 
            of this method is largely determined by m. 
            
            "OCT" is the Owen approach to compute the k-factor when controlling 
            the tails so that there is not more than (1-P)/2 of the data in each 
            tail of the distribution.
            
        The default is "HE"
        
    m : TYPE, optional
        The maximum number of subintervals to be used in the integrate 
        function. This is necessary only for method = "EXACT" and method = 
        "OCT". The larger the number, the more accurate the solution. Too low 
        of a value can result in an error. A large value can also cause the 
        function to be slow for method = "EXACT". The default is 50.

Returns
-------
    anovatol.int returns a list where each element is a dataframe 
    corresponding to each main effect (i.e., factor) tested in the ANOVA and 
    the rows of each data frame are the levels of that factor. The columns of 
    each data frame report the following:
        
        mean:
            The mean for that factor level.
        
        n:
            The effective sample size for that factor level.
        
        k:	
            The k-factor for constructing the respective factor level's 
            tolerance interval.
        
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
    
    Howe, W. G. (1969), Two-Sided Tolerance Limits for Normal Populations - 
        Some Improvements, Journal of the American Statistical Association, 
        64, 610–620.

    Weissberg, A. and Beatty, G. (1969), Tables of Tolerance Limit Factors 
        for Normal Distributions, Technometrics, 2, 483–500.

Examples
--------
    ## 90%/95% 2-sided tolerance intervals for a 2-way ANOVA 
    
    ## NOTE: Response must be the leftmost entry in dataframe and lm object
        breaks = '26 30 54 25 70 52 51 26 67 18 21 29 17 12 18 35 30 36 36 21 24 18 10 43 28 15 26 27 14 29 19 29 31 41 20 44 42 26 19 16 39 28 21 39 29 20 21 24 17 13 15 15 16 28'.split(" ")
        
        breaks = [float(a) for a in breaks]
        
        wool = 'A A A A A A A A A A A A A A A A A A A A A A A A A A A B B B B B B B B B B B B B B B B B B B B B B B B B B B'.split(' ')
        
        tension = 'L L L L L L L L L M M M M M M M M M H H H H H H H H H L L L L L L L L L M M M M M M M M M H H H H H H H H H'.split(' ')
        
        warpbreaks = pd.DataFrame({'breaks':breaks,'wool':wool,
                                   'tension':tension})
        
        lmout = ols('breaks ~ wool + tension',warpbreaks).fit()
        
        anovatolint(lmout, data = warpbreaks, alpha = 0.10, P = 0.95, side = 2
                    , method = "HE")
     
Note for When Using
        response variable y must be the leftmost object in the dataframe, the 
        first entered creating an lm object, 2 steps
            1.) df = pandas.DataFrame({'response':response, 'x1':x1, 'x2':x2,
                                       ...}))
            
            2.) ols('response ~ x1 + x2 +...', data = df).fit()

        data MUST be entered with response being first in lm and dataframe 
        (on the leftmost) it should only have a format with the y and x's 
        being in their place below ols(response ~ x1 + x2 + ..., data = df).fit()
        
    '''
    out = anova_lm(lmout)
    dim1 = len(out.iloc[:,0])-1
    s = np.sqrt(out.iloc[dim1][2])
    df = list(int(k) for k in out.iloc[:,0])
    xlev = levels(data,out)
    resp = data.columns[0] #gets the response variable, y
    #resp_ind = int(np.where(data.columns == resp)[0]) #should be 0
    #pred_ind = np.where(data.columns != resp)[0]
    factors = [a[0] for a in xlev]
    outlist = []
    bal = []
    lev = list([np.array(a[1]).ravel() for a in xlev])
    for i in range(len(factors)):
        tempmeans = []
        templens = []
        tempmeans_without_level = []
        templens_without_level = []
        templow = []
        tempup = []
        K = []
        for j in range(len(lev[i])):
           tempmeans.append([lev[i][j], np.mean(data[data[factors[i]] == lev[i][j]][resp])])
           templens.append([lev[i][j], length(data[data[factors[i]] == lev[i][j]][resp])])   
           K.append(Kfactor(n = templens[j][1],f = df[-1], alpha = alpha, P = P, side = side, method = method, m = m))
           templow.append(tempmeans[j][1]-K[j]*s)
           tempup.append(tempmeans[j][1]+K[j]*s)
           tempmeans_without_level.append(np.mean(data[data[factors[i]] == lev[i][j]][resp]))
           templens_without_level.append(length(data[data[factors[i]] == lev[i][j]][resp]))
           tempmat = pd.DataFrame({'temp.means':tempmeans_without_level,'temp.eff':templens_without_level, 'K':K, 'temp.low':templow, 'temp.up':tempup})
        tempmat.index = [lev[i]]
        if side == 1:
            tempmat.columns = ["mean", "n", "k", "1-sided.lower", "1-sided.upper"]
        else:
            tempmat.columns = ["mean", "n", "k", "2-sided.lower", "2-sided.upper"]
        outlist.append(tempmat)
        #print(tempmat)
        t = np.array([templen[1] for templen in templens])
        bal.append(np.where(sum(abs(t - np.mean(t))>3)))
        bal = [to_int(x) for x in bal]
    bal = sum(bal)
    if bal > 0:
        return "This procedure should only be used for balanced (or nearly-balanced) designs."
    if side == 1:
        print(f'These are {(1-alpha)*100}%/{P*100}% {side}-sided tolerance limits.')
    else:
        print(f'These are {(1-alpha)*100}%/{P*100}% {side}-sided tolerance intervals.')
    for i in range(length(outlist)):
        outlist[i] = outlist[i].sort_values(by=['mean'],ascending = False)
    fin = [[i[0] for i in xlev], [a for a in outlist]]      
    return dict(zip(fin[0],fin[1]))
   #for i in range(len(fin[1])):
    #    st += f'{fin[0][i]}\n{fin[1][i]}\n\n'
    #return st

import sympy as sp
import inspect

def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    return len(x)

def nonlinregtolint(formula, xydata, xnew = None, side = 1, alpha = 0.05, P = 0.99, maxiter = 100):
    '''
Nonlinear Regression Tolerance Bounds

Description
    Provides 1-sided or 2-sided nonlinear regression tolerance bounds.

Usage
    nlregtolint(formula, xydata, xnew = None, side = 1, alpha = 0.05, P = 0.99)
                
Parameters
----------
    formula: function
        A nonlinear model formula including variables and parameters.

    xydata: dataframe
        A data frame in which to evaluate the formulas in formula. The first 
        column of xydata must be the response variable.

    xnew: list or float, optional
        Any new levels of the predictor(s) for which to report the tolerance 
        bounds. The number of columns must be 1 less than the number of 
        columns for xydata.

    side: 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance bound is required 
        (determined by side = 1 or side = 2, respectively).

    alpha: float, optional
        The level chosen such that 1-alpha is the confidence level.

    P: float, optional
        The proportion of the population to be covered by the tolerance 
        bound(s).

    maxiter: int, optional
        A positive integer specifying the maximum number of iterations that 
        the nonlinear least squares routine (curve_fit) should run.

Details
    It is highly recommended that the user specify starting values for the 
    curve_fit routine.

Returns
-------
    nlregtolint returns a data frame with items:

        alpha	
            The specified significance level.

        P	
            The proportion of the population covered by the tolerance bound(s).

        yhat	
            The predicted value of the response for the fitted nonlinear 
            regression model.

        y	
            The value of the response given in the first column of xydata. 
            This data frame is sorted by this value.

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
        
    Wallis, W. A. (1951), Tolerance Intervals for Linear Regression, in Second 
        Berkeley Symposium on Mathematical Statistics and Probability, ed. J. 
        Neyman, Berkeley: University of CA Press, 43–51.

    Young, D. S. (2013), Regression Tolerance Intervals, Communications in 
        Statistics - Simulation and Computation, 42, 2040–2055.

Examples
    ## 95%/95% 2-sided nonlinear regression tolerance bounds for a sample of 
    size 50.
    
        np.random.seed(1)
        
        def formula1(x, b1, b2,b3):
            try:
                #make this the regular function using numpy
                
                return b1 + (0.49-b1)*np.exp(-b2*(x-8)) + b3**b3
            
            except:
                #make this the symbolic version of the function using sympy
                
                return b1 + (0.49-b1)*sp.exp(-b2*(x-8)) + b3**b3
    
        x = pd.DataFrame(st.uniform.rvs(size=50, loc=5, scale=45))
        
        y = formula1(x.iloc[:,0], 0.39, 0.11,0.01) + st.norm.rvs(size = length(x), scale = 0.01) #response
        
        xy = pd.concat([y,x],axis=1)
        
        xy.columns = ['y','x']
        
        nonlinregtolint(formula1, xydata=xy,alpha = 0.05, P = 0.95, side = 2)
    
    '''
    n = length(xydata.iloc[:,0])
    popt, pcov = opt.curve_fit(formula, xydata.iloc[:,1], xydata.iloc[:,0],maxfev=maxiter)
    residuals = xydata.iloc[:,0] - formula(xydata.iloc[:,1],*popt)
    ss_residuals = np.sum(residuals**2) #sum of squares residuals
    ms_residuals = ss_residuals/(n-2) #mean squares residuals
    try:
        sigma = np.sqrt(ms_residuals) #residual standard error
    except:
        return "Error in scipy.optimize.curve_fit(). Consider different staring estimates of the parameters."
    beta_hat = popt
    beta_names = inspect.getfullargspec(formula)[0][1:]
    x_name = inspect.getfullargspec(formula)[0][0]
    temp = pd.DataFrame(popt).T
    temp.columns = beta_names
    pars = length(beta_hat)
    bgroup = []
    for i in range(pars):
        temp = beta_names[i]
        bgroup.append(sp.Symbol(f'{temp}'))
    x = sp.Symbol(f'{x_name}')
    formula_prime_wrtb = []
    for i in range(pars):
        formula_prime_wrtb.append((sp.diff(formula(x, *bgroup),bgroup[i])))
    Pmat = [[]]*pars
    keys = [*bgroup]
    values = [*beta_hat]
    k = dict(zip(keys,values))
    for j in range(pars):
        sub = []
        for i in range(length(xydata.iloc[:,1])):
            sub.append(formula_prime_wrtb[j].subs({f'{x_name}':xydata.iloc[:,1][i],**k}))
        Pmat.append(sub)
    Pmat = [x for x in Pmat if x]
    Pmat = pd.DataFrame(Pmat).T
    PTP = np.dot(Pmat.T,Pmat).astype('float64')
    PTP2 = None
    PTP0 = PTP
    while PTP2 is None:
        try:
            PTP2 = np.linalg.inv(PTP)
        except:
            PTP3 = PTP0 + np.diag(np.linspace(min(np.diag(PTP))/1000,min(np.diag(PTP))/1000,length(np.diag(PTP))))
            try:
                PTPnew = np.linalg.inv(PTP3)
                PTP0 = PTP3
                PTP = PTPnew
            except:
                continue
        else:
            PTP = PTP2
    if xnew is not None:
        if length(xnew) == 1:
            xnew = pd.DataFrame(np.array([xnew]))
        else:
            xnew = pd.DataFrame(np.array(xnew))
        xtemp = pd.concat([pd.DataFrame([None,]*length(xnew)),xnew],axis=1)
        xtemp.columns = xydata.columns
        xydata = pd.concat([xydata,xtemp],axis = 0)
        xydata.index = range(length(xydata.iloc[:,1]))
        Pmat = [[]]*pars
        for j in range(pars):
            sub = []
            for i in range(length(xydata.iloc[:,1])):
                sub.append(formula_prime_wrtb[j].subs({f'{x_name}':xydata.iloc[:,1][i],**k}))
            Pmat.append(sub)
        Pmat = [x for x in Pmat if x]
        Pmat = pd.DataFrame(Pmat).T
    yhat = []
    for i in range(length(xydata.iloc[:,1])):
        yhat.append(formula(xydata.iloc[:,1][i],*beta_hat))
    nstar = [None,]*length(xydata.iloc[:,1])
    nrow = length(xydata.iloc[:,1])
    for i in range(nrow):
        nstar[i] = np.linalg.multi_dot([Pmat.iloc[i].T,PTP,Pmat.iloc[i].T.T])
    nstar = np.array(nstar)
    nstar = 1/nstar.astype('float64')
    df = n - pars
    if side == 1:
        zp = st.norm.ppf(P)
        delta = np.sqrt(nstar)*zp
        tdelta = st.nct.ppf(1-alpha, n - pars, nc = delta)
        tdelta[np.where(np.isnan(tdelta))] = np.inf
        K = tdelta/np.sqrt(nstar)
        K[np.where(np.isnan(K))] = np.inf
        upper = yhat + sigma*K
        lower = yhat - sigma*K
        temp = pd.DataFrame({"alpha":alpha, "P":P, "yhat":yhat, "y":xydata.iloc[:,0], "1-sided.lower":lower, "1-sided.upper":upper})
    else:
        K = np.sqrt(df*st.ncx2.ppf(P,1,1/nstar)/st.chi2.ppf(alpha,df))
        upper = yhat + sigma*K
        lower = yhat - sigma*K
        temp = pd.DataFrame({"alpha":alpha, "P":P, "yhat":yhat, "y":xydata.iloc[:,0], "2-sided.lower":lower, "2-sided.upper":upper})
    temp = temp.sort_values(by=['y'])
    temp.index = range(nrow)
    return temp



def regtolint(reg, DataFrame, newx = None, side = 1, alpha = 0.05, P = 0.99):
    '''
(Multiple) Linear Regression Tolerance Bounds

Description
    Provides 1-sided or 2-sided (multiple) linear regression tolerance bounds.
    It is also possible to fit a regression through the origin model.

Usage
    regtolint(reg, newx = None, side = 1, alpha = 0.05, P = 0.99) 
    
Parameters
----------
    reg: linear model
        An object of class 
        statsmodels.regression.linear_model.RegressionResultsWrapper
        (i.e., the results from a linear regression routine).
    
    DataFrame : DataFrame
        The DataFrame that holds all data.

    newx: DataFrame	
        An optional data frame in which to look for variables with which to 
        predict. If omitted, the fitted values are used.

    side: 1 or 2
        Whether a 1-sided or 2-sided tolerance bound is required (determined 
        by side = 1 or side = 2, respectively).

    alpha: float
        The level chosen such that 1-alpha is the confidence level.

    P: float
        The proportion of the population to be covered by the tolerance 
        bound(s).

Returns
-------
    regtolint returns a data frame with items:

        alpha:
            The specified significance level.

        P:
            The proportion of the population covered by the tolerance bound(s).

        y:	
            The value of the response given on the left-hand side of the model
            in reg.

        y.hat:
            The predicted value of the response for the fitted linear 
            regression model. This data frame is sorted by this value.

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
        
    Wallis, W. A. (1951), Tolerance Intervals for Linear Regression, in Second 
        Berkeley Symposium on Mathematical Statistics and Probability, ed. J. 
        Neyman, Berkeley: University of CA Press, 43–51.

    Young, D. S. (2013), Regression Tolerance Intervals, Communications in 
        Statistics - Simulation and Computation, 42, 2040–2055.
        
Examples
--------
    grain = pd.DataFrame([40, 17, 9, 15, 6, 12, 5, 9],columns = ['grain'])
    
    straw = pd.DataFrame([53, 19, 10, 29, 13, 27, 19, 30], columns = ['straw']) 
    
    df = pd.concat([grain,straw],axis = 1)
    
    newx = pd.DataFrame({'grain':[3,6,9]})
    
    reg = ols('straw ~ grain',data=df).fit()
    
    regtolint(reg, newx = newx,side=1)
    '''
    if side != 1 and side != 2:
        return "Must specify a one-sided or two-sided procedure!"
    try:
        reg.params
    except:
        return "Input must be of class statsmodels.regression.linear_model.RegressionResultsWrapper"
    else: 
        n = length(reg.resid)
        pars = length(reg.params)
        newlength = 0
        #est = reg.predict() #mvreg[i].predict(newx)
        e = reg.get_prediction().summary_frame()
        est_fit = e.iloc[:,0]
        est_sefit = e.iloc[:,1]
        est1_fit,est1_sefit = None,None
        if type(newx) == pd.core.frame.DataFrame:
            newlength = length(newx)
            e1 = reg.get_prediction(newx).summary_frame()
            est1_fit = e1.iloc[:,0]
            est1_sefit = e1.iloc[:,1]
            yhat = np.hstack([est_fit,est1_fit])
            sey = np.hstack([est_sefit, est1_sefit])
        else:
            yhat = est_fit
            sey = est_sefit
        y = np.hstack([DataFrame.iloc[:,1].values, np.array([None,]*newlength)])
        a_out = anova_lm(reg)
        MSE = a_out['mean_sq'][length(a_out['mean_sq'])-1]
        deg_freedom = int(a_out['df'][length(a_out['df'])-1])
        nstar = MSE/sey**2
        if side == 1:
            zp = st.norm.ppf(P)
            delta = np.sqrt(nstar) * zp
            tdelta = st.nct.ppf(1-alpha,df= int(n-pars),nc = delta)
            K = tdelta/np.sqrt(nstar)
            upper = yhat + np.sqrt(MSE)*K
            lower = yhat - np.sqrt(MSE)*K
            temp = pd.DataFrame({'alpha':alpha,'P':P,'y':y,'yhat':yhat,'1-sided.lower':lower,'1-sided.upper':upper})
        else:
            K = np.sqrt(deg_freedom*st.ncx2.ppf(P,1,1/nstar)/st.chi2.ppf(alpha,deg_freedom))
            upper = yhat + np.sqrt(MSE)*K
            lower = yhat - np.sqrt(MSE)*K
            temp = pd.DataFrame({'alpha':alpha,'P':P,'y':y,'yhat':yhat,'2-sided.lower':lower,'2-sided.upper':upper})
        temp = temp.sort_values(by=['yhat'])
        temp.index = range(length(temp.iloc[:,0]))
        return temp

def tricubic(x):
    y = np.zeros_like(x)
    idx = (x >= -1) & (x <= 1)
    y[idx] = np.power(1.0 - np.power(np.abs(x[idx]), 3), 3)
    return y

#nonparametric smoothing routine
#https://github.com/joaofig/pyloess
class Loess(object):

    @staticmethod
    def normalize_array(array):
        min_val = np.min(array)
        max_val = np.max(array)
        return (array - min_val) / (max_val - min_val), min_val, max_val

    def __init__(self, xx, yy, degree=1):
        self.n_xx, self.min_xx, self.max_xx = self.normalize_array(xx)
        self.n_yy, self.min_yy, self.max_yy = self.normalize_array(yy)
        self.degree = degree

    @staticmethod
    def get_min_range(distances, window):
        min_idx = np.argmin(distances)
        n = len(distances)
        if min_idx == 0:
            return np.arange(0, window)
        if min_idx == n-1:
            return np.arange(n - window, n)

        min_range = [min_idx]
        while len(min_range) < window:
            i0 = min_range[0]
            i1 = min_range[-1]
            if i0 == 0:
                min_range.append(i1 + 1)
            elif i1 == n-1:
                min_range.insert(0, i0 - 1)
            elif distances[i0-1] < distances[i1+1]:
                min_range.insert(0, i0 - 1)
            else:
                min_range.append(i1 + 1)
        return np.array(min_range)

    @staticmethod
    def get_weights(distances, min_range):
        max_distance = np.max(distances[min_range])
        weights = tricubic(distances[min_range] / max_distance)
        return weights

    def normalize_x(self, value):
        return (value - self.min_xx) / (self.max_xx - self.min_xx)

    def denormalize_y(self, value):
        return value * (self.max_yy - self.min_yy) + self.min_yy

    def estimate(self, x, window, use_matrix=False, degree=1):
        n_x = self.normalize_x(x)
        distances = np.abs(self.n_xx - n_x)
        min_range = self.get_min_range(distances, window)
        weights = self.get_weights(distances, min_range)

        if use_matrix or degree > 1:
            wm = np.multiply(np.eye(window), weights)
            xm = np.ones((window, degree + 1))

            xp = np.array([[math.pow(n_x, p)] for p in range(degree + 1)])
            for i in range(1, degree + 1):
                xm[:, i] = np.power(self.n_xx[min_range], i)

            ym = self.n_yy[min_range]
            xmt_wm = np.transpose(xm) @ wm
            beta = np.linalg.pinv(xmt_wm @ xm) @ xmt_wm @ ym
            y = (beta @ xp)[0]
        else:
            xx = self.n_xx[min_range]
            yy = self.n_yy[min_range]
            sum_weight = np.sum(weights)
            sum_weight_x = np.dot(xx, weights)
            sum_weight_y = np.dot(yy, weights)
            sum_weight_x2 = np.dot(np.multiply(xx, xx), weights)
            sum_weight_xy = np.dot(np.multiply(xx, yy), weights)

            mean_x = sum_weight_x / sum_weight
            mean_y = sum_weight_y / sum_weight

            b = (sum_weight_xy - mean_x * mean_y * sum_weight) / \
                (sum_weight_x2 - mean_x * mean_x * sum_weight)
            a = mean_y - b * mean_x
            y = a + b * n_x
        return self.denormalize_y(y)

def f(n,alpha,P):
    return alpha - (n * P**(n - 1) - (n - 1) * P**n)

def bisection(a,b,n,alpha,tol=1e-8):
    xl = a
    xr = b
    while np.abs(xl-xr) >= tol:
        c = (xl+xr)/2
        prod = f(n=n,alpha=alpha,P=xl)*f(n=n,alpha=alpha,P=c)
        if prod > tol:
            xl = c
        else:
            if prod < tol:
                xr = c
    return c

def distfreeest2(n = None, alpha = None, P = None, side = 1):
    temp = 0
    if n == None:
        temp += 1
    if alpha == None:
        temp +=1
    if P == None:
        temp += 1
    if temp > 1:
        return 'Must specify values for any two of n, alpha, and P'
    if (side != 1 and side != 2):
        return 'Must specify a 1-sided or 2-sided interval'
    if side == 1:
        if n == None:
            ret = int(np.ceil(np.log(alpha)/np.log(P)))
        if P == None:
            ret = np.exp(np.log(alpha)/n)
            ret = float(f'{ret:.4f}')
        if alpha == None:
            ret = 1-P**n
    else:
        if alpha == None:
            ret = 1-(np.ceil((n*P**(n-1)-(n-1)*P**n)*10000))/10000
        if n == None:
            ret = int(np.ceil(opt.brentq(f,a=0,b=1e100,args=(alpha,P),maxiter=1000)))
        if P == None:
            ret = np.ceil(bisection(0,1,alpha =alpha, n = n, tol = 1e-8)*10000)/10000    
    return ret

def distfreeest(n = None, alpha = None, P = None, side = 1):
    if n == None:
        if type(alpha) == float:
            alpha = [alpha]
        if type(P) == float:
            P = [P]
        A = length(alpha)
        B = length(P)
        column_names = np.zeros(B)
        row_names = np.zeros(A)
        matrix = np.zeros((A,B))
        for i in range(A): 
            row_names[i] = alpha[i]
            for j in range(B):
                column_names[j] = P[j]
                matrix[i,j] = distfreeest2(alpha=alpha[i],P=P[j],side=side)
        out = pd.DataFrame(matrix,columns = column_names, index = row_names)
        
    if alpha == None:
        if type(n) == float or type(n) == int:
            n = [n]
        if type(P) == float:
            P = [P]
        A = length(n)
        B = length(P)
        column_names = np.zeros(B)
        row_names = np.zeros(A)
        matrix = np.zeros((A,B))
        for i in range(A): 
            row_names[i] = n[i]
            for j in range(B):
                column_names[j] = P[j]
                matrix[i,j] = distfreeest2(n=n[i],P=P[j],side=side)
        out = pd.DataFrame(matrix,columns = column_names, index = row_names)
        
    if P == None:
        if type(alpha) == float:
            alpha = [alpha]
        if type(n) == float or type(n) == int:
            n = [n]
        A = length(alpha)
        B = length(n)
        print(f'length of alpha = {A}',f'length of n = {B}')
        column_names = np.zeros(B)
        row_names = np.zeros(A)
        matrix = np.zeros((A,B))
        for i in range(A): 
            row_names[i] = alpha[i]
            for j in range(B):
                column_names[j] = n[j]
                matrix[i,j] = distfreeest2(alpha=alpha[i],n=n[j],side=side)
        out = pd.DataFrame(matrix,columns = column_names, index = row_names)
    return out

def nptolint(x,alpha = 0.05, P = 0.99, side = 1, method = 'WILKS', upper = None, lower = None):
    '''
    nptolint(x,alpha = 0.05, P = 0.99, side = 1, method = ('WILKS','WALD','HM','YM'), upper = None, lower = None):
Parameters
----------
    x: list
        A vector of data which no distributional assumptions are made. 
        The data is only assumed to come from a continuous distribution.
    
    alpha: float, optional
        The level chosen such that 1-alpha is the confidence level. 
        The default is 0.05.
    
    P: float, optional
        The proportion of the population to be covered by this tolerance 
        interval. The default is 0.99.
    
    side: 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively). The default is 1.
    
    method: string, optional
        The method for determining which indices of the ordered observations will be used for the tolerance intervals. 
        "WILKS" is the Wilks method, which produces tolerance bounds symmetric 
        about the observed center of the data by using the beta distribution. 
        
        "WALD" is the Wald method, which produces (possibly) multiple tolerance 
        bounds for side = 2 (each having at least the specified confidence level), 
        but is the same as method = "WILKS" for side = 1. 
        
        "HM" is the Hahn-Meeker method, which is based on the binomial distribution, 
        but the upper and lower bounds may exceed the minimum and maximum of the sample data. 
        For side = 2, this method will yield two intervals if an odd number of 
        observations are to be trimmed from each side. 
        
        "YM" is the Young-Mathew method for performing interpolation or 
        extrapolation based on the order statistics. 
        See below for more information on this method.
        
        The default is "WILKS"
    
    upper: float, optional 
        The upper bound of the data. When None, then the maximum of x is used. 
        If method = "YM" and extrapolation is performed, then upper will be 
        greater than the maximum. The default value is None.
    
    lower: float, optional
        The lower bound of the data. When None, then the minimum of x is used. 
        If method = "YM" and extrapolation is performed, then lower will be 
        less than the minimum. The default value is None.
    
Details
    For the Young-Mathew (YM) method, interpolation or extrapolation is performed. 
    When side = 1, two intervals are given: one based on linear 
    interpolation/extrapolation of order statistics (OS-Based) and one based 
    on fractional order statistics (FOS-Based). When side = 2, only an interval 
    based on linear interpolation/extrapolation of order statistics is given.
    
Returns
-------
  nptolint returns a data frame with items:
        
    alpha: The specified significance level.
    
    P: The proportion of the population covered by this tolerance interval.
    
    1-sided.lower: The 1-sided lower tolerance bound. This is given only if side = 1.
    
    1-sided.upper: The 1-sided upper tolerance bound. This is given only if side = 1.
    
    2-sided.lower: The 2-sided lower tolerance bound. This is given only if side = 2.
    
    2-sided.upper: The 2-sided upper tolerance bound. This is given only if side = 2.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance Intervals. 
        Journal of Statistical Software, 36(5), 1-39. URL http://www.jstatsoft.org/v36/i05/.
    
    Bury, K. (1999), Statistical Distributions in Engineering, Cambridge University Press.

    Hahn, G. J. and Meeker, W. Q. (1991), Statistical Intervals: A Guide for 
        Practitioners, Wiley-Interscience.

    Wald, A. (1943), An Extension of Wilks' Method for Setting Tolerance Limits, 
        The Annals of Mathematical Statistics, 14, 45–55.

    Wilks, S. S. (1941), Determination of Sample Sizes for Setting Tolerance 
        Limits, The Annals of Mathematical Statistics, 12, 91–96.

    Young, D. S. and Mathew, T. (2014), Improved Nonparametric Tolerance 
        Intervals Based on Interpolated and Extrapolated Order Statistics, Journal 
        of Nonparametric Statistics, 26, 415–432.
Examples
--------
    ## 90%/95% 2-sided nonparametric tolerance intervals for a sample of size 20. 

    nptol.int(x = x, alpha = 0.10, P = 0.95, side = 1, method = "WILKS", upper = NULL, lower = NULL)
    '''
    n = len(x)
    if n < 2:
        return 'cannot do anything with less than 2 datapoints'
    xsort = np.sort(x)
    if(upper == None):
        upper = np.max(x)
    if(lower == None):
        lower = np.min(x)
    if(method == "WILKS"):
        if(side == 2):
            if(np.floor((n+1)/2) == (n+1)/2):
                up = ((n + 1)/2) - 1
            else:
                up = np.floor((n + 1)/2)
            r = np.arange(1,up+1)
            #r = np.array([1,2,3,4,.001,.002])
            out2 = 1-scipy.stats.beta.cdf(P, n - 2 * r + 1, 2 * r) - (1-alpha)
            ti2 =pd.DataFrame([r,out2])
            ti2 = ti2.T #transpose the dataframe to make it easier to work with
            temp2 = ti2[ti2[1]>0] #Gets all rows where col2 > 0
            if len(temp2) == 0:
                lower = lower
                upper = upper
            else:
                mins2 = min(temp2[1])
                temp2 = temp2[temp2[1]==mins2]
                r = int(temp2[0])
                lower = xsort[r]
                upper = xsort[n-r+1]
            d = {'alpha': [alpha], 'P': [P], '2-sided lower':lower, '2-sided upper':upper}
            temp = pd.DataFrame(data=d)
                
        if(side ==1):
            r = scipy.stats.binom.ppf(alpha, n= n, p=1-P)
            s = n-r+1
            if(r<1):
                lower = lower
            else:
                lower = xsort[int(r)]
            if (s > n):
                upper = upper
            else:
                upper = xsort[int(s-1)]
            d = {'alpha': [alpha], 'P': [P], '1-sided lower':lower, '1-sided upper':upper}
            temp = pd.DataFrame(data=d)
    if(method == "WALD"): #needs to be made more effient for side == 1 and side == 2
        t = []
        s = []
        for i in range(2,n+1):
            s.extend(list(range(1,i)))
            t.extend((i,)*(i-1))
            
        if side == 1: #Make this code more efficient
            r = scipy.stats.binom.ppf(alpha, n = n, p = 1-P)
            s = n-r+1
            if r < 1:
                lower = lower
            else:
                lower = xsort[int(r)]
            if s > n:
                upper = upper
            else:
                upper = xsort[int(s)]
            d = {'alpha': [alpha], 'P': [P], '1-sided lower':lower, '1-sided upper':upper}
            temp = pd.DataFrame(data = d)
        else: #Make this code more efficient
            out3 = []
            for i in range(len(t)):
                out3.append(1 - scipy.stats.beta.cdf(P, int(t[i]-s[i]), int(n-t[i]+s[i]+1))-(1-alpha))
            #ti3 = pd.DataFrame({'s':s,'t':t,'out3':out3}).T
            ti3 =pd.DataFrame([s,t,out3])
            ti3 = ti3.T #transpose the dataframe to make it easier to work with
            temp3 = ti3[ti3[2] > 0] #should be >
            if len(temp3) == 0:
                lower = lower
                upper = upper
            else:
                mins3 = min(temp3[2])
                out5 = temp3[temp3[2]==mins3]
                s = out5[0]
                t = out5[1]
                s = s.tolist()
                t = t.tolist()
                for i in range(len(s)):
                    t[i] = t[i]-1
                    s[i] = s[i]-1
                lower = np.zeros(len(s))
                upper = np.zeros(len(s))
                for i in range(len(t)):
                    lower[i] = xsort[int(s[i])]
                    upper[i] = xsort[int(t[i])]
            if length(lower) == 1 and length(upper) == 1:
                d = {'alpha': [alpha], 'P': [P], '2-sided lower':[lower], '2-sided upper':[upper]}
                temp = pd.DataFrame(data = d)
            else:
                d = {'alpha': [alpha], 'P': [P], '2-sided lower':lower[0], '2-sided upper':upper[0]}
                d = pd.DataFrame(data=d)
                for i in range(1,len(lower)):
                    d.loc[len(d.index)] = [alpha,P,lower[i],upper[i]]
                temp = d
    if (method == 'HM'):
        ind = range(n+1)
        out = scipy.stats.binom.cdf(ind, n = n, p = P) - (1-alpha)
        ti = pd.DataFrame([ind,out]).T
        temp = ti[ti[1] > 0]
        mins = min(temp[1])
        HMind = int(temp[temp[1] == mins][0])
        diff = n - HMind
        if side == 2:
            if diff == 0 or int(np.floor(diff/2)) == 0:
                if lower != None:
                    xsort = np.insert(xsort, 0, lower)#pd.DataFrame([lower, xsort])
                if upper != None:
                    xsort = np.insert(xsort, len(xsort), upper)#pd.DataFrame([xsort, upper]) #come back to this area when done
                HM = [1] + [len(xsort)]                
                d = {'alpha': [alpha], 'P': [P], '2-sided lower': lower, '2-sided upper': upper}#xsort[int(HM.loc[0][1])]}
                temp = pd.DataFrame(data=d)
            else:
                if np.floor(diff/2) == diff/2:
                    v1 = diff/2 #scalar
                    v2 = diff/2
                else:
                    v1 = [np.floor(diff/2), np.ceil(diff/2)] #list
                    v2 = [sum(x) for x in zip(v1, [1,-1])] #add v1 to [1,-1] element-wise
                if type(v1) == list:
                    #you can make this block more effient
                    data = {'v1': [v1[0]], 'v2*': [n- v2[0] +1]}
                    HM = pd.DataFrame(data = data)
                    for i in range(1,len(v1)):
                        HM.loc[len(HM.index)] = [v1[i], n-v2[i]+1]
                    
                    d = {'alpha': [alpha], 'P': [P], '2-sided lower': xsort[int(HM.loc[0][0])], '2-sided upper': xsort[int(HM.loc[0][1]-1)]}#xsort[int(HM.loc[0][1])]}
                    d = pd.DataFrame(data = d)
                    for i in range(1,len(HM)):
                        d.loc[len(d.index)] = [alpha,P,xsort[int(HM.loc[i][0])],xsort[int(HM.loc[i][1]-1)]]    
                    temp = d
                else:
                    data = {'v1': [v1], 'v2*': [n-v2+1]}
                    HM = pd.DataFrame(data = data)
                    d = {'alpha': [alpha], 'P': [P], '2-sided lower': xsort[int(HM.loc[0][0])], '2-sided upper': xsort[int(HM.loc[0][1])-1]}
                    temp = pd.DataFrame(data = d)
                if len(HM) == 2 and len(HM.loc[0]) == 2: #is the row dim 2 and col dim 2? T/F
                    if xsort[int(HM.loc[0][0])] == xsort[int(HM.loc[1][0])] and xsort[int(HM.loc[0][1])-1] == xsort[int(HM.loc[1][1])-1]:
                        temp = temp.loc[0]
                        temp = pd.DataFrame(temp).T
        else:
            l = pd.DataFrame(range(n+1),columns=['l'])
            lp = pd.DataFrame((1-scipy.stats.binom.cdf(l-1,n,1-P))-(1-alpha),columns=[''])
            lowtemp = pd.concat([l,lp],axis=1)
            u =  pd.DataFrame(range(1,n+2),columns=['u'])
            up = pd.DataFrame((scipy.stats.binom.cdf(u-1,n,P))-(1-alpha),columns=[''])
            uptemp = pd.concat([u,up],axis=1)
            l = lowtemp[lowtemp.loc[:,'']>0]
            l = max(l.loc[:,'l'])
            if l > 0:
                lower = xsort[l-1]
            u = uptemp[uptemp.loc[:,'']>0]
            u = min(u.loc[:,'u'])
            if u < n+1:
                upper = xsort[u-1]
            d = {'alpha': [alpha], 'P': [P], '1-sided lower': lower, '1-sided upper': upper}
            temp = pd.DataFrame(data=d)
    if method == 'YM':
        nmin = int(np.array(distfreeest(alpha = alpha, P=P, side = side).iloc[0]))
        temp = None
        if side == 1:
            if n < nmin:
                temp = extrap(x=x,alpha=alpha,P=P)
            else:
                temp = interp(x=x,alpha=alpha,P=P)
        else:
            temp = twosided(x=x,alpha=alpha,P=P)
    return temp

def fl(u1,u,n,alpha):
    #error of 1e-5 compared to R
    return scipy.stats.beta.cdf(u,(n+1)*u1,(n+1)*(1-u1))-1+alpha

def fu(u2,u,n,alpha):
    #error of 1e-5 compared to R
    return scipy.stats.beta.cdf(u,(n+1)*u2,(n+1)*(1-u2))-alpha

def eps(u,n):
    return (n+1)*u-np.floor((n+1)*u)

def neps(u,n):
    return -((n+1)*u-np.floor((n+1)*u))

def LSReg(y,x,gamma):  
    
    xbar = sum(x)/len(x)
    ybar = sum(y)/len(y)
    sumx = []
    sumx2 = []
    sumy = []
    sumxy = []
    for i in range(len(x)):
        sumx.append(x[i])
        sumx2.append(x[i]**2)
        sumy.append(y[i])
        sumxy.append(y[i]*x[i])
    ssxx = sum(sumx2) - (1/len(x))*sum(sumx)**2
    ssxy = sum(sumxy) - (1/len(y))*sum(sumy)*sum(sumx)
    B1 = ssxy/ssxx
    B0 = ybar - B1*xbar
    return B0+B1*gamma #regression equation with xi = gamma      

def interp(x,alpha,P):
    n = len(x)
    x = sorted(x)
    gamma = 1-alpha
    out = list(nptolint(range(n+1),alpha=alpha,P=P,side=1)[['1-sided lower','1-sided upper']].loc[0])
    s = out[0]
    r = out[1]
    ###Beran-Hall
    pil = (gamma-scipy.stats.binom.cdf(n-s-1,n,P))/scipy.stats.binom.pmf(n-s,n,P)
    piu = (gamma-scipy.stats.binom.cdf(r-2,n,P))/scipy.stats.binom.pmf(r-1,n,P)
    if s == n:
        Ql = x[s]
    else:
        Ql = pil*x[s+1]+(1-pil)*x[s]
    if r == 1:
        Qu = x[r-1]
    else:
        Qu = piu*x[r-1] + (1-piu)*x[r-2]
    ###Hutson
    u1 = opt.brentq(fl, a = 0.00001,b=0.99999,args=(1-P,n,alpha))
    u2 = opt.brentq(fu, a = 0.00001,b=0.99999,args=(P,n,alpha))
    if s == n:
        Qhl = x[s]
    else:
        Qhl = (1 - eps(u1, n)) * x[s] + eps(u1, n) * x[s + 1]
    if r == 1:
        Qhu = x[r-1]
    else:
        Qhu = (1 - eps(u2, n)) * x[r - 2] + eps(u2, n) * x[r-1]
    names = ['alpha','P','1-sided.lower','1-sided.upper']
    temp = pd.DataFrame([[alpha,P,Ql,Qu],[alpha,P,Qhl,Qhu]],columns=names)
    temp.index = ['OS-Based','FOS-Based']
    #return value is slightly different than R due to rounding. and scipy.stats.beta.cdf() vs pbeta()
    return temp

def extrap(x,alpha,P):
    n = len(x)
    x = sorted(x)
    gamma = 1-alpha
    out = list(nptolint(range(n+1),alpha=alpha,P=P,side=1)[['1-sided lower','1-sided upper']].loc[0])
    pib = -(gamma-scipy.stats.binom.cdf(n-1,n,P))/(scipy.stats.binom.pmf(n-1,n,P))
    Qexpl = pib*x[1]+(1-pib)*x[0]
    Qexpu = pib*x[n-2]+(1-pib)*x[n-1]
    u1b = opt.brentq(fl, a = 0.00001,b=0.99999,args=(1-P,n,alpha))
    u2b = opt.brentq(fu, a = 0.00001,b=0.99999,args=(P,n,alpha))
    Qhexpl = (1-neps(u1b,n))*x[0]+neps(u1b,n)*x[1]
    Qhexpu = (1-neps(u2b,n))*x[n-1]+neps(u2b,n)*x[n-2]
    names = ['alpha','P','1-sided.lower','1-sided.upper']
    temp = pd.DataFrame([[alpha,P,Qexpl,Qexpu],[alpha,P,Qhexpl,Qhexpu]],columns=names)
    temp.index = ['OS-Based','FOS-Based']
    return temp

def twosided(x,alpha,P):
    n = len(x)
    x = sorted(x)
    gamma = 1-alpha
    out = nptolint(range(n+1),alpha=alpha,P=P,side=2,method='HM')[['2-sided lower','2-sided upper']]
    r = np.ravel(np.array(out[['2-sided lower']]).T)
    s = np.ravel(np.array(out[['2-sided upper']]).T)
    r = [int(x) for x in r]
    s = [int(x) for x in s]
    if (len(out.index) == 2): #around 430,000 datapoints needed for this to be true
        X1L = np.array([x[r[0]],x[r[0]+1]])
        X2L = np.array([x[r[1]],x[r[1]+1]])
        X1U = np.array([x[s[0]],x[s[0]-1]])
        X2U = np.array([x[s[1]],x[s[1]-1]])
        g = np.ravel(np.array([(scipy.stats.binom.cdf(s[0]-r[0]-1,n,P),(scipy.stats.binom.cdf(s[0]-(r[0]+1)-1,n,P)))]))
        #predict using X1L and g, you are here
        out1L = LSReg(X1L,g,gamma)
        out2L = LSReg(X2L,g,gamma)
        out1U = LSReg(X1U,g,gamma)
        out2U = LSReg(X2U,g,gamma)
        temp1 = pd.DataFrame({'0':[out1L,out2L,x[r[0]],x[r[1]]]})
        temp2 = pd.DataFrame({'1':[x[s[0]],x[s[1]],out1U,out2U]})
        temp3 = pd.DataFrame({'2':[x[s[0]]-out1L,x[s[1]]-out2L,out1U-x[r[0]],out2U-x[r[0]]]})
        temp = pd.concat([temp1,temp2,temp3],axis=1)
        if scipy.stats.binom.cdf(s[1]-r[1]-1,n,P) >= gamma:
            indtemp = list(temp['2'])
            ind = indtemp.index(max(indtemp))
            temp = list(temp.iloc[ind,0:2])
            if ind==1 or ind==3:
                ord1 = 1
            else:
                indtemp = list(temp['2'])
                ind = indtemp.index(max(indtemp))
                temp = list(temp.iloc[ind,0:2])
                if ind==1 or ind ==3:
                    ord1 = 1
                else:
                    ord1 = 2
    else:
        XL = np.array([x[r[0]],x[r[0]+1]])
        if s[0] == length(x):
            XU = np.array([x[s[0]-1],x[s[0]-2]])
            print(s[0]-(r[0]+1)-1)
            g = np.ravel(np.array([(scipy.stats.binom.cdf(s[0]-(r[0]+1)-1,n,P)),(scipy.stats.binom.cdf(s[0]-(r[0]+1)-2,n,P))]))
            print(g)
            outL = LSReg(XL,g,gamma)
            outU = LSReg(XU,g,gamma)
            temp1 = pd.DataFrame({'0':[outL,x[r[0]]]})
            temp2 = pd.DataFrame({'1':[x[s[0]-1],outU]})
            temp3 = pd.DataFrame({'2':[x[s[0]-1]-outL,outU-x[r[0]]]})
            temp = pd.concat([temp1,temp2,temp3],axis=1)
        else:
            XU = np.array([x[s[0]],x[s[0]+1]])
            g = np.ravel(np.array([(scipy.stats.binom.cdf(s[0]-r[0]-1,n,P),(scipy.stats.binom.cdf(s[0]-(r[0]+1)-1,n,P)))]))
            outL = LSReg(XL,g,gamma)
            outU = LSReg(XU,g,gamma)
            temp1 = pd.DataFrame({'0':[outL,x[r[0]]]})
            temp2 = pd.DataFrame({'1':[x[s[0]],outU]})
            temp3 = pd.DataFrame({'2':[x[s[0]]-outL,outU-x[r[0]]]})
            temp = pd.concat([temp1,temp2,temp3],axis=1)
        if scipy.stats.binom.cdf(s[0]-r[0]-1,n,P) >= gamma:
            indtemp = list(temp['2'])
            ind = indtemp.index(min(indtemp))
            temp = list(temp.iloc[ind,0:2])
        else:
            temp = list([outL,outU])
    temp = pd.DataFrame({'alpha':[alpha], 'P':[P],'2-sided.lower':temp[0],'2-sided.upper':temp[1]},['OS-Based'])
    return temp

def npregtolint(x, y, yhat, side = 1, alpha = 0.05, P = 0.99, method = 'WILKS', upper = None, lower = None):
    '''
Nonparametric Regression Tolerance Bounds

Description
    Provides 1-sided or 2-sided nonparametric regression tolerance bounds.

Usage
    npregtolint(x, y, yhat, side = 1, alpha = 0.05, P = 0.99,
                method = ["WILKS", "WALD", "HM"], upper = None, 
                lower = None)

Parameters
----------
    x : array
        A vector of values for the predictor variable. Currently, this 
        function is only capable of handling a single vector.

    y : array
        A vector of values for the response variable.
        
    yhat : array
        A vector of fitted values extracted from a nonparametric smoothing 
        routine. 
        
    side : 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance bound is required (determined 
        by side = 1 or side = 2, respectively). The default is 1.
        
    alpha : float, optional
        The level chosen such that 1-alpha is the confidence level. The 
        default is 0.05.
        
    P : float, optional
        The proportion of the population to be covered by the tolerance 
        bound(s). The default is 0.99.
        
    method : string, optional
        The method for determining which indices of the ordered residuals will
        be used for the tolerance bounds. "WILKS", "WALD", and "HM" are each 
        described in nptolint. However, since only one tolerance bound can 
        actually be reported for this procedure, only the first tolerance 
        bound will be returned. Note that this is not an issue when method = 
        "WILKS" is used as it only produces one set of tolerance bounds. The 
        default is 'WILKS'.
        
    upper : float, optional
        The upper bound of the data. When None, then the maximum of x is used. 
        The default is None.
        
    lower : float, optional
        The lower bound of the data. When None, then the minimum of x is used. The default is None.

Returns
-------
    npregtolint returns a data frame with items:

        alpha	
            The specified significance level.

        P	
            The proportion of the population covered by the tolerance bound(s).

        x	
            The values of the predictor variable.

        y	
            The values of the response variable.

        y.hat	
            The predicted value of the response for the fitted nonparametric 
            smoothing routine.
                
        1-sided.lower	
            The 1-sided lower tolerance bound. This is given only if side = 1.

        1-sided.upper	
            The 1-sided upper tolerance bound. This is given only if side = 1.

        2-sided.lower	
            The 2-sided lower tolerance bound. This is given only if side = 2.

        2-sided.upper	
            The 2-sided upper tolerance bound. This is given only if side = 2.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Young, D. S. (2013), Regression Tolerance Intervals, Communications in 
        Statistics - Simulation and Computation, 42, 2040–2055.
        
Examples
--------
    ## 95%/99% 1-sided nonparametric regression tolerance bounds for a sample
    of size 16.
    
        x = np.array([5,10,12,7,40,27,12,30,22,32,44,9,17,25,33,12])
        
        def f(x):
            return x**(1.2345)
        
        y = f(x)
        
        loess = Loess(x,y)
        
        yhat = []
        
        for a in x:
            
            yhat.append(loess.estimate(a, window = 8, use_matrix = False, 
                                       degree = 2))
        
        npregtolint(x, y, yhat)
    '''
    n = length(x)
    if length(x) != n or length(y) != n or length(yhat) != n:
        return "The predictor vector, response vector, and fitted value vector must all be of the same length!"
    if length(x) == 1:
        x = [x]
        y = [y]
        yhat = [yhat]
    x = np.array(x)
    y = np.array(y)
    yhat = np.array(yhat)
    res = y-yhat
    toltemp = nptolint(res, side = side, alpha = alpha, P = P, method = method, upper = upper, lower = lower)
    outtemp = []
    upper = []
    lower = []
    temp = []
    for i in (range(length(toltemp.iloc[:,0]))):
        upper.append(yhat + toltemp.iloc[i,3])
        lower.append(yhat + toltemp.iloc[i,2])
        alpha = pd.DataFrame([alpha,]*length(x))
        P = pd.DataFrame([P,]*length(x))
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        yhat = pd.DataFrame(yhat)
        lower = pd.DataFrame(lower).T
        upper = pd.DataFrame(upper).T
        temp = pd.concat([alpha,P,x,y,yhat,lower,upper],axis=1)
        if side == 1:
            temp.columns = ["alpha", "P", "x", "y", "yhat", "1-sided.lower", "1-sided.upper"]
        else:
            temp.columns = ["alpha", "P", "x", "y", "yhat", "2-sided.lower", "2-sided.upper"]
        index = int(np.where(temp.columns == 'yhat')[0])
        temp = temp.sort_values(by='x')
        temp.index = (range(length(x)))
        outtemp.append(temp)
    if length(outtemp) == 1:
        outtemp = outtemp[0]
    return outtemp

# Need this function to run the 3d graph in plottol
# https://github.com/CircusMonkey/covariance-ellipsoid/blob/master/ellipsoid.py
def get_cov_ellipsoid(cov, mu=np.zeros((3)), nstd=3):
    """
    Return the 3d points representing the covariance matrix
    cov centred at mu and scaled by the factor nstd.
    Plot on your favourite 3d axis. 
    Example 1:  ax.plot_wireframe(X,Y,Z,alpha=0.1)
    Example 2:  ax.plot_surface(X,Y,Z,alpha=0.1)
    """
    assert cov.shape==(3,3)
    # Find and sort eigenvalues to correspond to the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.sum(cov,axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    # Set of all spherical angles to draw our ellipsoid
    n_points = 100
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Width, height and depth of ellipsoid
    rx, ry, rz = nstd * np.sqrt(eigvals)

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    X = rx * np.outer(np.cos(theta), np.sin(phi))
    Y = ry * np.outer(np.sin(theta), np.sin(phi))
    Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Rotate ellipsoid for off axis alignment
    old_shape = X.shape
    # Flatten to vectorise rotation
    X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
    X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
    X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)
   
    # Add in offsets for the mean
    X = X + mu[0]
    Y = Y + mu[1]
    Z = Z + mu[2]
    return X,Y,Z

    plt.show()

def plottol(tolout, xdata = [1], y = None, side = 1, NonLinReg = False, xlab = 'X', ylab = 'Y', zlab = 'Z',title = None):
    '''
Plotting Capabilities for Tolerance Intervals

Description
    Provides control charts and/or histograms for tolerance bounds on 
    continuous data as well as tolerance ellipses for data distributed 
    according to bivariate and trivariate normal distributions. Scatterplots 
    with regression tolerance bounds and interval plots for ANOVA tolerance 
    intervals may also be produced.

Usage
    plottol(tolout, x, y = None, side = 1)
    
Parameters
----------
    tolout: dataframe	
        Output from any continuous (including ANOVA) tolerance interval 
        procedure or from a regression tolerance bound procedure.

    xdata: list
        Either data from a continuous distribution or the predictors for a 
        regression model. If this is a design matrix for a linear regression 
        model, then it must be in matrix form AND include a column of 1's if 
        there is to be an intercept. 
        
    y: list, optional
        The response vector for a regression setting. Leave as None if not 
        doing regression tolerance bounds.

    side: 1 or 2, optional
        side = 2 produces plots for either the two-sided tolerance intervals 
        or both one-sided tolerance intervals. This will be determined by the 
        output in tolout. side = 1 produces plots showing the upper tolerance 
        bounds and lower tolerance bounds.
    
    NonLinReg: bool, optional
        True if user wants to do nonlinear regression, otherwise False.
    
    anova: dictionary, optional
        Dictionary with multiple ANOVA outputs from the anovatolint function.

Value
    plottol can return a control chart, histogram, or both for continuous data
    along with the calculated tolerance intervals. For regression data,
    plottol returns a scatterplot along with the regression tolerance bounds. 

References
    Montgomery, D. C. (2005), Introduction to Statistical Quality Control, 
        Fifth Edition, John Wiley & Sons, Inc.

Examples
    #1D        
    
    xdata = np.random.normal(size = 100)
    
    # # Example tolerance dataframe
    
    tol = pd.DataFrame([0.01, 0.95, 0.0006668252,-1.9643623,1.965696]).T
    
    tol.columns = ['alpha','P','mean','2-sided.lower','2-sided.upper']
    
    plottol(tol,xdata)
    
    #2D
    
    xdata = [np.random.normal(size = 100,loc=0,scale = 0.2), np.random.normal(size = 100,loc=0,scale = 0.5), np.random.normal(size = 100,loc=5,scale = 1)]
    
    # # Example tolerance dataframe
    
    tol = pd.DataFrame([7.383685]).T
    
    tol.columns = [0.01]
    
    tol.index = [0.99]
    
    plottol(tol,xdata)
    
    #Regression
    
    x = np.random.uniform(10, size = 100)
    
    y = 20 + 5*x+np.random.normal(3,size =100)
    
    data = pd.concat([pd.DataFrame(x),pd.DataFrame(y)],axis = 1)
    
    out = regtolint(reg = ols('y~x',data = data).fit(), DataFrame = data, side = 2, alpha = 0.05, P = 0.95)
    
    plottol(tolout = out, xdata=x, y=y)
    
    ## Nonlinear Regression
    
    np.random.seed(1)
    
    def formula1(x, b1, b2,b3):
        try:
            #make this the regular function using numpy
            
            return b1 + (0.49-b1)*np.exp(-b2*(x-8)) + b3**b3
        
        except:
            #make this the symbolic version of the function using sympy
            
            return b1 + (0.49-b1)*sp.exp(-b2*(x-8)) + b3**b3
    
    x = pd.DataFrame(st.uniform.rvs(size=50, loc=5, scale=45))
    
    y = formula1(x.iloc[:,0], 0.39, 0.11,0.01) + st.norm.rvs(size = length(x), scale = 0.01) #response
    
    xy = pd.concat([y,x],axis=1)
    
    xy.columns = ['y','x']
    
    YLIM = nonlinregtolint(formula1, xydata=xy,alpha = 0.05, P = 0.99, side = 2)
    
    plottol(YLIM,xdata=x,y=y,side=1,formula=formula1)
    
    #ANOVA 
    
    breaks = ('26 30 54 25 70 52 51 26 67 18 21 29 17 12 18 35 30 36 36 21 24 18 10 43 28 15 26 27 14 29 19 29 31 41 20 44 42 26 19 16 39 28 21 39 29 20 21 24 17 13 15 15 16 28'.split(" "))
    
    breaks = [float(a) for a in breaks]
    
    wool = 'A A A A A A A A A A A A A A A A A A A A A A A A A A A B B B B B B B B B B B B B B B B B B B B B B B B B B B'.split(' ')
    
    tension = 'L L L L L L L L L M M M M M M M M M H H H H H H H H H L L L L L L L L L M M M M M M M M M H H H H H H H H H'.split(' ')
    
    warpbreaks = pd.DataFrame({'breaks':breaks,'wool':wool,'tension':tension})
    
    lmout = ols('breaks ~ wool + tension',warpbreaks).fit()
    
    anova = anovatolint(lmout, data = warpbreaks, alpha = 0.10, P = 0.95, side = 2, method = "HE")
    
    plottol(anova)
    '''
    if type(xdata) is pd.core.frame.DataFrame:
        xdata = np.array(xdata)
    if type(xdata) is list:
        xdata = np.array(xdata)
    if xdata.ndim == 1 and y is None and NonLinReg == False and type(tolout) is not dict:
        if '2-sided.lower' in tolout.columns:
            tollower = tolout.iloc[:,tolout.columns.get_loc('2-sided.lower')][0]
        if '2-sided.upper' in tolout.columns:
            tolupper = tolout.iloc[:,tolout.columns.get_loc('2-sided.upper')][0]
        if '1-sided.lower' in tolout.columns:
            print("NOTE: The plot reflects two 1-sided tolerance intervals and NOT a 2-sided tolerance interval!")
            tollower = tolout.iloc[:,tolout.columns.get_loc('1-sided.lower')][0]
        if '1-sided.upper' in tolout.columns:
            tolupper = tolout.iloc[:,tolout.columns.get_loc('1-sided.upper')][0]
        fig, axs = plt.subplots(1,2)
        if '1-sided.lower' in tolout.columns:
            try:
                fig.suptitle(f"1-Sided {(1-tolout.iloc[:,tolout.columns.get_loc('alpha')][0])*100}%/{tolout.iloc[:,tolout.columns.get_loc('P')][0]*100}% Tolerance Limits")
            except:
                fig.suptitle(f"1-Sided (Beta = {(tolout.iloc[:,tolout.columns.get_loc('beta')][0])}) Limit")
        else:
            try:
                fig.suptitle(f"2-Sided {(1-tolout.iloc[:,tolout.columns.get_loc('alpha')][0])*100}%/{tolout.iloc[:,tolout.columns.get_loc('P')][0]*100}% Tolerance Limits")
            except:
                fig.suptitle(f"2-Sided (Beta = {(tolout.iloc[:,tolout.columns.get_loc('beta')][0])}) Limit")
        xs = np.arange(0,length(xdata),1)
        axs[0].scatter(xs,xdata)
        axs[0].axhline(tolupper,color = 'r',ls='dashed',label = 'upper limit')
        axs[0].axhline(tollower,color = 'r',ls='dashdot',label = 'lower limit')
        axs[1].hist(xdata)
        axs[1].axvline(tolupper,color = 'r',ls='dashed',label = 'upper limit')
        axs[1].axvline(tollower,color = 'r',ls='dashdot',label = 'lower limit')
        axs[1].legend(loc = 0,title = "Limits", bbox_to_anchor=(1.04, 1))
    elif length(xdata) == 2 and NonLinReg == False and type(tolout) is not dict:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        #ax.set_xlim([min(xdata[0]),max(xdata[0])])
        #ax.set_ylim([min(xdata[1]),max(xdata[1])])
        ax.scatter(xdata[0],xdata[1],s=10)
        #title = f"{(1-tolout.columns[0])*100}%/{tolout.index[0]*100}% Tolerance Region"
        ax.set_title(title)
        mu = xdata.mean(axis=1)
        sigma = np.cov(xdata)
        es = np.linalg.eigh(sigma)
        evals = es[0][::-1]
        evecs = es[1]
        evecs = [e[::-1] for e in evecs]
        e1 = np.dot(evecs,np.diag(np.sqrt(evals)))
        theta = np.linspace(0,2*np.pi, 1000)
        r1 = np.sqrt(tolout.values[0][0])
        v1 = pd.concat([pd.DataFrame(r1*np.cos(theta)),pd.DataFrame(r1*np.sin(theta))],axis=1)
        mu = np.expand_dims(mu,axis=-1)
        pts = pd.DataFrame((mu - np.dot(e1,v1.T)).T)
        #ax.autoscale(enable=True)
        ax.set_aspect('equal','datalim')
        ax.plot(pts[0],pts[1],color='r',lw=0.5)
    elif length(xdata) == 3 and NonLinReg == False and type(tolout) is not dict:
        fig = plt.figure()
        ax = fig.add_subplot(111,projection = '3d')
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_zlabel(zlab)
        # ax.set_xlim([min(xdata[0]),max(xdata[0])])
        # ax.set_ylim([min(xdata[1]),max(xdata[1])])
        # ax.set_zlim([min(xdata[2]),max(xdata[2])])
        ax.autoscale(enable=True)
        ax.scatter3D(xdata[0],xdata[1],xdata[2],c=xdata[2])
        title = f"{(1-tolout.columns[0])*100}%/{tolout.index[0]*100}% Tolerance Region"
        ax.set_title(title)
        Mean = xdata.mean(axis=1)
        Sigma = np.cov(xdata)
        X,Y,Z = get_cov_ellipsoid(Sigma, Mean, nstd=np.sqrt(tolout.values[0][0]))
        ax.plot_surface(X, Y, Z, color='r',alpha = 0.2)
        plt.show()
    elif y is not None and length(y) > 2 and NonLinReg == False and type(tolout) is not dict:
        print("This only works when the forumula is a single linear model of the form y = b0 + b1*x")
        plt.scatter(xdata,y,color = 'black', linewidths = 1)
        df = pd.concat([pd.DataFrame(xdata),pd.DataFrame(y)],axis=1)
        lm = ols('y~xdata',data=df).fit()
        u = np.linspace(min(xdata),max(xdata),1000)
        fu = lm.params[0] + lm.params[1]*u
        plt.plot(u,fu,color = 'black')
        lmw = ols('tolout.iloc[:,4]~sorted(xdata)', data = pd.concat([tolout.iloc[:,4],pd.DataFrame(xdata)],axis = 1)).fit()
        w = np.linspace(min(xdata),max(xdata),1000)
        fw = lmw.params[0] + lmw.params[1]*w
        lmx = ols('tolout.iloc[:,5]~sorted(xdata)', data = pd.concat([tolout.iloc[:,5],pd.DataFrame(xdata)],axis = 1)).fit()
        xx = np.linspace(min(xdata),max(xdata),1000)
        fxx = lmx.params[0] + lmx.params[1]*xx
        plt.plot(xx,fxx, color = 'r', label = "Upper Limit", ls = ':')
        plt.plot(w,fw, color = 'r', label = 'Lower Limit',ls='dashed')
        plt.title(f"{(1-tolout.iloc[0,0])*100}%/{tolout.iloc[0,1]*100}% Tolerance Limits")
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.show()
    elif NonLinReg and type(tolout) is not dict:
        xdata = pd.DataFrame(xdata)
        out1 = pd.concat([xdata,pd.DataFrame(y)],axis=1)
        out1.columns = ['XX','YY']
        out1 = out1.sort_values(by=['YY'])
        out1.index = range(length(out1.iloc[:,0]))
        out1 = pd.concat([out1.iloc[:,0],tolout.iloc[:length(xdata)]],axis=1)
        out1 = out1.sort_values(by=['XX'])
        out1.index = range(length(out1.iloc[:,0]))
        if length(tolout.iloc[0]) == 6:
            if '2-sided.lower' in tolout.columns:
                plt.plot(out1.iloc[:,0],out1.iloc[:,5], color = 'r', ls='-.', label = '2-Sided Lower Limit')
                plt.plot(out1.iloc[:,0],out1.iloc[:,3], color = 'black', ls='-', label = 'Best Fit Line')
                plt.plot(out1.iloc[:,0],out1.iloc[:,6], color = 'r', ls='--', label = '2-Sided Upper Limit')
            else:
                print("NOTE: The plot reflects two 1-sided tolerance intervals and NOT a 2-sided tolerance interval!")
                plt.plot(out1.iloc[:,0],out1.iloc[:,5], color = 'r', ls='-.', label = '1-Sided Lower Limit')
                plt.plot(out1.iloc[:,0],out1.iloc[:,3], color = 'black', ls='-', label = 'Best Fit Line')
                plt.plot(out1.iloc[:,0],out1.iloc[:,6], color = 'r', ls='--', label = '1-Sided Upper Limit')
        elif length(tolout.iloc[0]) == 7:
            if '1-sided.lower' in tolout.columns:
                print("NOTE: The plot reflects two 1-sided tolerance intervals and NOT a 2-sided tolerance interval!")
                plt.plot(out1.iloc[:,0],out1.iloc[:,6], color = 'r', ls='-.', label = '1-Sided Lower Limit')
                plt.plot(out1.iloc[:,0],out1.iloc[:,5], color = 'black', ls='-', label = 'Best Fit Line')
                plt.plot(out1.iloc[:,0],out1.iloc[:,7], color = 'r', ls='--', label = '1-Sided Upper Limit')
            else:
                plt.plot(out1.iloc[:,0],out1.iloc[:,6], color = 'r', ls='-.', label = '2-Sided Lower Limit')
                plt.plot(out1.iloc[:,0],out1.iloc[:,5], color = 'black', ls='-', label = 'Best Fit Line')
                plt.plot(out1.iloc[:,0],out1.iloc[:,7], color = 'r', ls='--', label = '2-Sided Upper Limit')
        plt.title(f"{(1-tolout.iloc[0,0])*100}%/{tolout.iloc[0,1]*100}% Tolerance Limits")
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.scatter(xdata.iloc[:,0],y)
        plt.legend(loc = 0, bbox_to_anchor=(1.04, 1))
        plt.show()
    elif type(tolout) is dict:
        keys = []
        values = []
        for i in range(length(tolout)):
            keys.append(list(tolout)[i])
            values.append(list(tolout.values())[i])
        if '1-sided.lower' in values[0]:
            print("NOTE: The plot reflects two 1-sided tolerance intervals and NOT a 2-sided tolerance interval!")
        dictlen = length(tolout)
        fig, axs = plt.subplots(1,dictlen)
        A1ticks = []
        for i in range(dictlen):
            tmpticks = []
            for j in range(length(values[i].iloc[:,0])):
                tmpticks.append(values[i].index.tolist()[j][0])
            A1ticks.append(tmpticks)
            axs[i].set_title(f'{keys[i]}')
            axs[i].set_ylim([np.min(values[i].values)-abs(np.min(values[i].values)/10),np.max(values[i].values)+abs(np.max(values[i].values)/10)])
            axs[i].set_xlim([0.5,length(values[i].iloc[:,0])+0.5])
            axs[i].set_xticks(list(range(1,length(values[i].iloc[:,0])+1)))
            axs[i].set_xticklabels(A1ticks[i])
        for j in range(dictlen):
            for i in range(length(values[j].iloc[:,0])):
                axs[j].scatter(i+1,values[j].iloc[i,0])
        ranges = []
        ymins = []
        ymaxs = []
        for i in range(dictlen):
            ranges.append(length(values[i].iloc[:,0]))
            tmpymin = []
            tmpymax = []
            for j in range(length(values[i].iloc[:,0])):
                tmpymin.append(values[i].iloc[j,3])
                tmpymax.append(values[i].iloc[j,4])
            ymins.append(tmpymin)
            ymaxs.append(tmpymax)
        ranges = [list(range(1,rans + 1)) for rans in ranges]
        for i in range(dictlen):
            axs[i].vlines(x=ranges[i],ymin=ymins[i],ymax=ymaxs[i],color = 'r',lw=0.5)

        
# ANOVA
# data equivalent to warpbreaks in R
# breaks = ('26 30 54 25 70 52 51 26 67 18 21 29 17 12 18 35 30 36 36 21 24 18 10 43 28 15 26 27 14 29 19 29 31 41 20 44 42 26 19 16 39 28 21 39 29 20 21 24 17 13 15 15 16 28'.split(" "))
# breaks = [float(a) for a in breaks]
# wool = 'A A A A A A A A A A A A A A A A A A A A A A A A A A A B B B B B B B B B B B B B B B B B B B B B B B B B B B'.split(' ')
# tension = 'L L L L L L L L L M M M M M M M M M H H H H H H H H H L L L L L L L L L M M M M M M M M M H H H H H H H H H'.split(' ')
# warpbreaks = pd.DataFrame({'breaks':breaks,'wool':wool,'tension':tension})
# lmout = ols('breaks ~ wool + tension',warpbreaks).fit()
# anova = anovatolint(lmout, data = warpbreaks, alpha = 0.10, P = 0.95, side = 1, method = "HE")
# plottol(anova)
      
## Nonlinear regression
    ## Example 1
# np.random.seed(1)
# def formula1(x, b1, b2):
#     try:
#         #make this the regular function using numpy
#         return b1 + (0.49-b1)*np.exp(-b2*(x-8))
#     except:
#         #make this the symbolic version of the function using sympy
#         return b1 + (0.49-b1)*sp.exp(-b2*(x-8)) 
# x = pd.DataFrame(st.uniform.rvs(size=500, loc=5, scale=45))
# #x = pd.DataFrame(np.array([44.5, 1.1, 6.2, 35.2, 23.8, 30.1, 13.9]))
# y = formula1(x.iloc[:,0], 0.39, 0.11) + st.norm.rvs(size = length(x), scale = 0.01) #response
# #print(y)
# #y = pd.Series(np.array([.38,.58, .54, .37, .43, .39, .44]))
# xy = pd.concat([y,x],axis=1)
# xy.columns = ['y','x']
# YLIM = nonlinregtolint(formula1, xydata=xy,alpha = 0.05, P = 0.99, side = 2)
# print(YLIM)
# plottol(YLIM,xdata=x,y=y,side=1,NonLinReg = True)
    ## Example 2
# x = np.array([5,10,12,7,40,27,12,30,22,32,44,9,17,25,33,12])
# def formula1(x,b):
#     return b*x**(1.2345)
# y = formula1(x,0.5) + scipy.stats.norm.rvs(size = 16, scale = 3)
# loess = Loess(x,y)
# yhat = []
# for a in x:
#     yhat.append(loess.estimate(a, window = 8, use_matrix = False, 
#                                 degree = 2))
# YLIM = npregtolint(x, y, yhat)
# plottol(YLIM,xdata=x,y=y,side=1,NonLinReg = True)
        
#1D        
    ## Example 1
# xdata = np.random.normal(size = 100)
# # Example tolerance dataframe
# tol = pd.DataFrame([0.01, 0.95, 0.0006668252,-1.9643623,1.965696]).T
# tol.columns = ['alpha','P','mean','2-sided.lower','2-sided.upper']
# plottol(tol,xdata)
    ## Example 2
# xdata = st.cauchy.rvs(size = 1000, loc = 100000, scale = 10)
# # # Example tolerance dataframe
# tol = pd.DataFrame([0.05, 0.9,99931.11,100067.9]).T
# tol.columns = ['alpha','P','2-sided.lower','2-sided.upper']
# plottol(tol,xdata)
    ## Example 3
# xdata = st.expon.rvs(size =100)
# print(xdata)
# tol = pd.DataFrame([0.9,0.006914525,0.5901049]).T
# tol.columns = ['beta','2-sided.lower','2-sided.upper']
# plottol(tol,xdata)



# # 3D
# xdata = [np.random.normal(size = 100,loc=0,scale = 0.2), np.random.normal(size = 100,loc=0,scale = 0.5), np.random.normal(size = 100,loc=5,scale = 1)]
# # Example tolerance dataframe
# tol = pd.DataFrame([7.383685]).T
# tol.columns = [0.1]
# tol.index = [0.9]
# plottol(tol,xdata)
# np.random.seed(1)
# x1 = np.random.normal(0,0.2,size = 1000)
# x2 = np.random.normal(0,0.5,size = 1000)
# x3 = np.random.normal(5,1,size = 1000)
# xdata = [x1,x2,x3]
# tol = pd.DataFrame([11.814179]).T
# tol.columns = [0.01]
# tol.index = [0.99]
# plottol(tol,xdata)

# #2D
#np.random.seed(1)
# x1 = np.random.normal(0,0.2,size = 1000)
# x2 = np.random.normal(0,0.5,size = 1000)
# #RFDAWG
# def generate_TLE(typed,meanx=0,meany=0,sdx=1,sdy=1,size = 50):
#     #pandas is slow
#     if typed == 'normal':
#         cross = np.random.normal(meanx,sdx,size)
#         along = np.random.normal(meany,sdy,size)
#         return [cross,along]
#     if typed == 'cluster': #same method to fix this as 'two'
#         cross = np.random.normal(meanx,sdx,int(size*.8))
#         along = np.random.normal(meany,sdy,int(size*.8))
#         cross = np.hstack((cross,np.random.normal(100,50,int(size*.2)))) #meanx* = 100, sdx*=50
#         along = np.hstack((along,np.random.normal(2000,50,int(size*.2)))) #meany* = 2000, sdy* = 50
#         return [cross,along]
#     if typed == 'two':
#         cross = np.random.normal(meanx,sdx,int(size*.5))
#         along = np.random.normal(meany,sdy,int(size*.5))
#         cross = np.hstack((cross,np.random.normal(500,100,int(size*.5)))) #meanx* = 500, sdx*=100
#         along = np.hstack((along,np.random.normal(1500,500,int(size*.5)))) #meany* = 1500, sdy* = 500
#         return [cross, along]
#     if typed == 'outliers':
#         cross = np.random.normal(meanx,sdx,int(size*.90))
#         along = np.random.normal(meany,sdy,int(size*.90))
#         cross = np.hstack((cross,np.random.normal(0,500,int(size*.1))))
#         along = np.hstack((along,np.random.normal(0,2500,int(size*.1))))
#         return [cross,along]
#     else:
#         return 'Incorrect type specified'
# x1, x2 = generate_TLE('normal',meanx=0,meany=0,sdx=100,sdy=500,size = 100)
# xdata = [x1,x2]
# tol = pd.DataFrame([4.653130553857355]).T
# tol.columns = [0.1] #alpha
# tol.index = [0.9] #P
# plottol(tol,xdata,xlab = 'Cross-Track Error (ft)', ylab = 'Along-Track Error (ft)', title = 'normal')

# x1, x2 = generate_TLE('two',meanx=0,meany=0,sdx=100,sdy=500,size = 100)
# xdata = [x1,x2]
# tol = pd.DataFrame([4.653215991165444]).T
# tol.columns = [0.1] #alpha
# tol.index = [0.9] #P
# plottol(tol,xdata,xlab = 'Cross-Track Error (ft)', ylab = 'Along-Track Error (ft)', title = 'two')

# x1, x2 = generate_TLE('cluster',meanx=0,meany=0,sdx=100,sdy=500,size = 100)
# xdata = [x1,x2]
# tol = pd.DataFrame([4.652875519340747]).T
# tol.columns = [0.1] #alpha
# tol.index = [0.9] #P
# plottol(tol,xdata,xlab = 'Cross-Track Error (ft)', ylab = 'Along-Track Error (ft)', title = 'cluster')

# x1, x2 = generate_TLE('outliers',meanx=0,meany=0,sdx=100,sdy=500,size = 100)
# xdata = [x1,x2]
# tol = pd.DataFrame([4.6544325019075075]).T
# tol.columns = [0.1] #alpha
# tol.index = [0.9] #P
# plottol(tol,xdata,xlab = 'Cross-Track Error (ft)', ylab = 'Along-Track Error (ft)', title = 'outliers')


# #Linear Regression
#x = np.random.uniform(10, size = 5)
#y = 20 + 5*x+np.random.normal(3,size =5)
# data = pd.concat([pd.DataFrame(x),pd.DataFrame(y)],axis = 1)
# out = regtolint(reg = ols('y~x',data = data).fit(), DataFrame = data, side = 2, alpha = 0.05, P = 0.95)
# plottol(tolout = out, xdata=x, y=y, xlab = 'Explanatory', ylab='Response')