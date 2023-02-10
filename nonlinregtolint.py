import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.optimize as opt
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
