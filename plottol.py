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
                    print("The ellison method should only be used for f appreciably larger than n^2")
                    return None
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
                    return integrate.quad(Fun1,a = f1 * delta**2/(ke**2 * n), b = 1000 * n, args=(P,ke,n,f1,delta),limit = m)
                def Fun3(ke,P,n,f1,alpha,m,delta):
                    f = Fun2(ke = ke, P = P, n = n, f1 = f1, alpha = alpha, m = m, delta = delta)
                    return abs(f[0] - (1-alpha))
                K = opt.minimize(fun=Fun3, x0=k2, args=(P,n,f,alpha,m,delta), method = 'L-BFGS-B')['x']
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
        #TEMP = np.vectorize(Ktemp)
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
    fin = [[i[0] for i in xlev], [a for a in outlist]]
    st = ''         
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

def plottol(tolout, xdata = [1], y = None, side = 1, formula = None):
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
    
    formula: ols object, optional
        A nonlinear model formula including variables and parameters.
    
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
    if xdata.ndim == 1 and y is None and formula is None and type(tolout) is not dict:
        if '2-sided.lower' in tolout.columns:
            tollower = tolout.iloc[:,tolout.columns.get_loc('2-sided.lower')][0]
        if '2-sided.upper' in tolout.columns:
            tolupper = tolout.iloc[:,tolout.columns.get_loc('2-sided.upper')][0]
        if '1-sided.lower' in tolout.columns:
            tollower = tolout.iloc[:,tolout.columns.get_loc('1-sided.lower')][0]
        if '1-sided.upper' in tolout.columns:
            tolupper = tolout.iloc[:,tolout.columns.get_loc('1-sided.upper')][0]
        fig, axs = plt.subplots(1,2)
        if '1-sided.lower' in tolout.columns:
            fig.suptitle(f"1-Sided {(1-tolout.iloc[0,0])*100}%/{tolout.iloc[0,1]}% Tolerance Limits")
        else:
            fig.suptitle(f"2-Sided {(1-tolout.iloc[0,0])*100}%/{tolout.iloc[0,1]*100}% Tolerance Limits")
        xs = np.arange(0,length(xdata),1)
        axs[0].scatter(xs,xdata)
        axs[0].axhline(tolupper,color = 'r',ls='dashed',label = 'upper limit')
        axs[0].axhline(tollower,color = 'r',ls='dashdot',label = 'lower limit')
        axs[1].hist(xdata)
        axs[1].axvline(tolupper,color = 'r',ls='dashed',label = 'upper limit')
        axs[1].axvline(tollower,color = 'r',ls='dashdot',label = 'lower limit')
        axs[1].legend(loc = 0,title = "Limits", bbox_to_anchor=(1.04, 1))
    
    elif xdata.ndim == 2 and formula is None and type(tolout) is not dict:
        if '1-sided.lower' in tolout.columns:
            side = 1
        elif '2-sided.lower' in tolout.columns:
            side = 2
        else:
            side = side 
        if side == 2:
            xup = normtolint(xdata[0],side = side,alpha = tolout.columns[0], P = tolout.index[0])
            xup = xup.iloc[0,xup.columns.get_loc('2-sided.upper')]
            yup = normtolint(xdata[1],side = side,alpha = tolout.columns[0], P = tolout.index[0])
            yup = yup.iloc[0,yup.columns.get_loc('2-sided.upper')]
        else:
            xup = normtolint(xdata[0],side = side,alpha = tolout.columns[0], P = tolout.index[0])
            xup = xup.iloc[0,xup.columns.get_loc('1-sided.upper')]
            yup = normtolint(xdata[1],side = side,alpha = tolout.columns[0], P = tolout.index[0])
            yup = yup.iloc[0,yup.columns.get_loc('1-sided.upper')]
        fig = plt.figure()
        ax = fig.add_subplot(111,projection = '3d')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim([min(xdata[0]),max(xdata[0])])
        ax.set_ylim([min(xdata[1]),max(xdata[1])])
        ax.set_zlim([min(xdata[2]),max(xdata[2])])
        ax.scatter3D(xdata[0],xdata[1],xdata[2],c=xdata[2])
        title = f"{(1-tolout.columns[0])*100}%/{tolout.index[0]*100}% Tolerance Region"
        ax.set_title(title)
        Mean = xdata.mean(axis=1)
        phi = np.linspace(0,2*np.pi, 256).reshape(256, 1) # the angle of the projection in the xy-plane
        theta = np.linspace(0, np.pi, 256).reshape(-1, 256) # the angle from the polar axis, ie the polar angle
        # Transformation formulae for a spherical coordinate system.
        x = Mean[0]+xup*np.sin(theta)*np.cos(phi)
        y = Mean[1]+yup*np.sin(theta)*np.sin(phi)
        z = Mean[2]+np.sqrt(tolout.values)*np.cos(theta)
        ax.plot_surface(x, y, z, color='r',alpha = 0.2)
        plt.show()
    elif y is not None and length(y) > 2 and formula is None and type(tolout) is not dict:
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
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    elif formula is not None and type(tolout) is not dict:
        xdata = pd.DataFrame(xdata)
        popt, pcov = curve_fit(formula, xdata=sorted(xdata.iloc[:,0],reverse=True), ydata=tolout.iloc[:,4],maxfev=2000)
        plt.plot(sorted(xdata.iloc[:,0]),formula(np.sort(xdata.iloc[:,0]), *popt),color = 'r',ls='--', label = 'Lower Limit')
        popt, pcov = curve_fit(formula, xdata=xdata.iloc[:,0], ydata=y)
        plt.plot(sorted(xdata.iloc[:,0]),formula(np.sort(xdata.iloc[:,0]), *popt),color = 'black',ls='-', label = 'Best Fit Line')
        popt, pcov = curve_fit(formula, xdata=sorted(xdata.iloc[:,0],reverse=True), ydata=tolout.iloc[:,5],maxfev=2000)
        plt.plot(sorted(xdata.iloc[:,0]),formula(np.sort(xdata.iloc[:,0]), *popt),color = 'r',ls='-.',label = "Upper Limit")
        plt.title(f"{(1-tolout.iloc[0,0])*100}%/{tolout.iloc[0,1]*100}% Tolerance Limits")
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.scatter(xdata.iloc[:,0],y)
        plt.legend(loc = 0, bbox_to_anchor=(1.04, 1))
        plt.show()
    elif type(tolout) is dict:
        keys = []
        values = []
        for i in range(length(tolout)):
            keys.append(list(tolout)[i])
            values.append(list(tolout.values())[i])
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

            
        
        
                
        
        
            
        
        
        
    
# #ANOVA
# #data equivalent to warpbreaks in R
# breaks = ('26 30 54 25 70 52 51 26 67 18 21 29 17 12 18 35 30 36 36 21 24 18 10 43 28 15 26 27 14 29 19 29 31 41 20 44 42 26 19 16 39 28 21 39 29 20 21 24 17 13 15 15 16 28'.split(" "))
# breaks = [float(a) for a in breaks]
# wool = 'A A A A A A A A A A A A A A A A A A A A A A A A A A A B B B B B B B B B B B B B B B B B B B B B B B B B B B'.split(' ')
# tension = 'L L L L L L L L L M M M M M M M M M H H H H H H H H H L L L L L L L L L M M M M M M M M M H H H H H H H H H'.split(' ')
# warpbreaks = pd.DataFrame({'breaks':breaks,'wool':wool,'tension':tension})
# lmout = ols('breaks ~ wool + tension',warpbreaks).fit()
# anova = anovatolint(lmout, data = warpbreaks, alpha = 0.10, P = 0.95, side = 2, method = "HE")
# plottol(anova)
      
# ## Nonlinear regression
# np.random.seed(1)
# def formula1(x, b1, b2,b3):
#     try:
#         #make this the regular function using numpy
#         return b1 + (0.49-b1)*np.exp(-b2*(x-8)) + b3**b3
#     except:
#         #make this the symbolic version of the function using sympy
#         return b1 + (0.49-b1)*sp.exp(-b2*(x-8)) + b3**b3

# x = pd.DataFrame(st.uniform.rvs(size=50, loc=5, scale=45))
# y = formula1(x.iloc[:,0], 0.39, 0.11,0.01) + st.norm.rvs(size = length(x), scale = 0.01) #response
# xy = pd.concat([y,x],axis=1)
# xy.columns = ['y','x']
# YLIM = nonlinregtolint(formula1, xydata=xy,alpha = 0.05, P = 0.99, side = 2)
# plottol(YLIM,xdata=x,y=y,side=1,formula=formula1)
        
# #1D        
# xdata = np.random.normal(size = 100)
# # Example tolerance dataframe
# tol = pd.DataFrame([0.01, 0.95, 0.0006668252,-1.9643623,1.965696]).T
# tol.columns = ['alpha','P','mean','2-sided.lower','2-sided.upper']
# plottol(tol,xdata)

# # 2D
# xdata = [np.random.normal(size = 100,loc=0,scale = 0.2), np.random.normal(size = 100,loc=0,scale = 0.5), np.random.normal(size = 100,loc=5,scale = 1)]
# # Example tolerance dataframe
# tol = pd.DataFrame([7.383685]).T
# tol.columns = [0.1]
# tol.index = [0.9]
# plottol(tol,xdata)

# #Regression
# x = np.random.uniform(10, size = 100)
# y = 20 + 5*x+np.random.normal(3,size =100)
# data = pd.concat([pd.DataFrame(x),pd.DataFrame(y)],axis = 1)
# out = regtolint(reg = ols('y~x',data = data).fit(), DataFrame = data, side = 2, alpha = 0.05, P = 0.95)
# plottol(tolout = out, xdata=x, y=y)