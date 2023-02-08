import pandas as pd
import numpy as np
import scipy.stats as st
import scipy.optimize as opt


def logistolint(x, alpha = 0.05, P = 0.99, loglog = False, side = 1):
    '''
Logistic (or Log-Logistic) Tolerance Intervals

Description
    Provides 1-sided or 2-sided tolerance intervals for data distributed 
    according to a logistic or log-logistic distribution.

Usage
    logistolint(x, alpha = 0.05, P = 0.99, loglog = False, side = 1)
Parameters
----------
    x: list
        A vector of data which is distributed according to a logistic or 
        log-logistic distribution.

    alpha: float, optional
        The level chosen such that 1-alpha is the confidence level.

    P: float, optional
        The proportion of the population to be covered by this tolerance interval.

    loglog: bool, optional
        If True, then the data is considered to be from a log-logistic 
        distribution, in which case the output gives tolerance intervals for 
        the log-logistic distribution. The default is False.

    side: 1 or 2
        Whether a 1-sided or 2-sided tolerance interval is required 
        (determined by side = 1 or side = 2, respectively).

Details
    Recall that if the random variable X is distributed according to a 
    log-logistic distribution, then the random variable Y = ln(X) is 
    distributed according to a logistic distribution.

Returns
-------
    logistolint returns a data frame with items:

        alpha	
            The specified significance level.

        P	
            The proportion of the population covered by this tolerance 
            interval.

        1-sided.lower	
            The 1-sided lower tolerance bound. This is given only if side = 1.

        1-sided.upper	
            The 1-sided upper tolerance bound. This is given only if side = 1.

        2-sided.lower	
            The 2-sided lower tolerance bound. This is given only if side = 2.

        2-sided.upper	
            The 2-sided upper tolerance bound. This is given only if side = 2.

Note
----
    More data is ideal. More data means that the results from Python and R 
    become increasingly similar.

References
    Balakrishnan, N. (1992), Handbook of the Logistic Distribution, Marcel 
        Dekker, Inc.
    
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.

    Hall, I. J. (1975), One-Sided Tolerance Limits for a Logistic Distribution 
        Based on Censored Samples, Biometrics, 31, 873–880.

Examples
    ## 90%/95% 1-sided logistic tolerance intervals for a sample of size 20. 
    
    ## Ex. 1
        x = '3.0216954  5.1475711  7.7617662  7.0539224  7.3112302  3.6046474  6.9611540  6.5425018  2.2990498  6.3052713  4.2775977  2.8950655  3.2165382  4.1798766  3.7840294  3.6504121  6.5161855 -0.5594826  3.5656579  5.3668183'
        
        x = x.split(' ')
        
        x = np.array(x)
        
        x = x[np.where(x!='')]
        
        x = [float(x) for x in x]
        
        logistolint(x,side = 1)
        
    ## Ex. 2
        y = '4.8124826  6.1840660  4.8879357  6.0516887  4.5130188  1.0832841  7.5312747  7.0786679  5.3014504  5.8418427  4.6305065  7.5347813  4.5212669  4.8634318  6.7228232  5.8994346  2.3574882  3.0606887  5.2309173  4.2817307  5.0793965  5.3417808  4.8408976  3.6873539  4.8431411  4.0888879  5.9500094  5.2511237  3.9145767  4.5576780  3.7346115  6.5583443  5.5991345  4.3076442  3.8890053  3.6938511  5.5418210  5.0696426  6.2174327  4.7567477  7.9850456  5.3668726  7.0789399  0.4829771  5.0840224  4.1212985  6.2706668 -1.0448621  4.6580866  3.5755498  5.4251450 -0.2213298  6.2236163  3.8354886 11.1191975  6.1207386  6.0653569  6.2328771  2.8640310  7.0864997  2.3379200  4.3926794  5.0429086  5.8400923  6.5640883  3.7428032  5.8038282  5.5471172  4.3827777  7.1254492  5.9702492  7.1687094  4.0925794  3.7630224  5.0705065  4.5075937  4.8219902  5.4392266  4.5664842  3.7871812  4.3772991  2.7387124  7.6829035  4.9834132  2.8988407  6.9377542  3.7482882  4.9414070  3.7629936  8.4870809  9.8241016  4.3579277  5.6546435  7.3498010  5.8507000  7.9224707  3.4088450  0.8604575  2.8638904  3.8276065'
        
        y = y.split(' ')
        
        y = np.array(y)
        
        y = y[np.where(y!='')]
        
        y = [float(y) for y in y]
        
        logistolint(y,side = 1)
        
    ## Ex. 3
        np.random.seed(seed=1000)
        
        logistolint(st.logistic.rvs(size = 100, loc = 5, scale = 0.01),side = 1)
    '''
    if len(x) == 2:
        print("Warning: Bad approximation due to minimal data.")
    if side != 1 and side != 2:
        return 'Must specify a one-sided or two-sided procedure'
    if loglog:
        x = np.log(x)
    if side == 2:
        alpha = alpha/2
        P = (P+1)/2
    x = np.array(x)
    mmom = np.mean(x)
    smom = np.sqrt(3*(np.mean(x**2)-mmom**2))/np.pi
    x = list(x)
    inits = [mmom,smom]
    def logll(pars,x):
        return sum(-st.logistic.logpdf(x,loc = pars[0],scale = pars[1]))
    try:
        out = opt.minimize(logll, x0 = inits, args = (x), method = 'BFGS')
    except:
        L = mmom
        U = mmom
    else:
        out_est = out['x']
        m = out_est[0]
        s = out_est[1]
        invfish = out['hess_inv']
        var_m = invfish[0,0]
        var_s = invfish[1,1]
        covms = invfish[0,1]
        kdelta = st.logistic.ppf(P, scale = np.sqrt(3)/np.pi)
        t1 = kdelta - covms * st.norm.ppf(1-alpha)**2
        t2 = kdelta - covms * st.norm.ppf(1-alpha)**2
        u = kdelta**2 - var_m * st.norm.ppf(1-alpha)**2
        v = 1 - var_s * st.norm.ppf(1-alpha)**2
        klower = (t1+np.sqrt(t1**2-u*v))/v
        kupper = (t2+np.sqrt(t1**2-u*v))/v
        L = m - klower * s * np.pi/np.sqrt(3)
        U = m + kupper * s * np.pi/np.sqrt(3)
    if loglog:
        L = np.exp(L)
        U = np.exp(U)
    if side == 2:
        alpha = 2*alpha
        P = (2*P)-1
    temp = pd.DataFrame({"alpha":[alpha], "P":[P], "2-sided.lower":L, "2-sided.upper":U})
    if side == 1:
        temp.columns = ["alpha", "P", "1-sided.lower", "1-sided.upper"]
    return temp
    
log