#%matplotlib qt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.stats as st
import scipy.stats 
import scipy.integrate as integrate
import numpy as np
import scipy.optimize as opt
import pandas as pd
import warnings
import statistics
warnings.filterwarnings('ignore')

import scipy.stats 
import scipy.integrate as integrate
import numpy as np
import scipy.optimize as opt
import pandas as pd
import warnings
import statistics
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

def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    return len(x)



def plottol(tolout, xdata, y = None, side = 1):
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

    x: list
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

Note
----
    The code cannot handle anova tolerance intervals or polynomial regression
    intervals yet. 

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
    
    #ANOVA - not built yet
    
    '''
    if type(xdata) is pd.core.frame.DataFrame:
        xdata = np.array(xdata)
    if type(xdata) is list:
        xdata = np.array(xdata)
    if xdata.ndim == 1 and y is None:
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
    
    elif xdata.ndim == 2:
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
    elif y is not None and length(y) > 2:
        print("This only works when the forumula is a single linear model of he form y = b0 + b1*x")
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
        plt.legend()
        plt.show()
    
        
        
        
#1D        
# xdata = np.random.normal(size = 100)
# # Example tolerance dataframe
# tol = pd.DataFrame([0.01, 0.95, 0.0006668252,-1.9643623,1.965696]).T
# tol.columns = ['alpha','P','mean','2-sided.lower','2-sided.upper']
# plottol(tol,xdata)

#2D
# xdata = [np.random.normal(size = 100,loc=0,scale = 0.2), np.random.normal(size = 100,loc=0,scale = 0.5), np.random.normal(size = 100,loc=5,scale = 1)]
# # Example tolerance dataframe
# tol = pd.DataFrame([7.383685]).T
# tol.columns = [0.01]
# tol.index = [0.99]
# plottol(tol,xdata)

#Regression
# x = np.random.uniform(10, size = 100)
# y = 20 + 5*x+np.random.normal(3,size =100)
# data = pd.concat([pd.DataFrame(x),pd.DataFrame(y)],axis = 1)
# out = regtolint(reg = ols('y~x',data = data).fit(), DataFrame = data, side = 2, alpha = 0.05, P = 0.95)
# print(out)
# plottol(tolout = out, xdata=x, y=y)