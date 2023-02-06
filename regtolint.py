import scipy.stats as st
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pylab 
import scipy.stats as stats

def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    return len(x)


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
        
    
# grain = pd.DataFrame([40, 17, 9, 15, 6, 12, 5, 9],columns = ['grain'])
# straw = pd.DataFrame([53, 19, 10, 29, 13, 27, 19, 30], columns = ['straw']) 
# df = pd.concat([grain,straw],axis = 1)
# newx = pd.DataFrame({'grain':[3,6,9]})
# reg = ols('straw ~ grain',data=df).fit()
# print(regtolint(reg, DataFrame = df, newx = None, side=1))