import numpy as np
import pandas as pd
import scipy.stats 
import scipy.integrate as integrate
import scipy.optimize as opt
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

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
                print('This method prodcues slightly different results',
                      'when compared to R. Take these results with',
                      'a grain of salt. The range of error',
                      'is between approximately (1e-3,2.0).',
                      'If this method is abosolutely needed',
                      'use R instead.')
                def fun1(z,df1,P,X,n):
                    k = (scipy.stats.chi2.sf(df1*scipy.stats.chi2.ppf(P,1,z**2)/X**2,df=df1)*np.exp(-0.5*n*z**2))
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

# get ANOVA table as R like output

def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    return len(x)


# Ordinary Least Squares (OLS) model
#model = ols('value ~ C(treatments)', data=df_melt).fit()
#anova_table = sm.stats.anova_lm(model, typ=2)
#anova_table
# output (ANOVA F and p value)
#                sum_sq    df         F    PR(>F)
#C(treatments)  3010.95   3.0  17.49281  0.000026
#Residual        918.00  16.0       NaN       NaN

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
    for i in range(len(fin[1])):
        st += f'{fin[0][i]}\n{fin[1][i]}\n\n'
    return st
    
    
# #data equivalent to warpbreaks in R
# breaks = ('26 30 54 25 70 52 51 26 67 18 21 29 17 12 18 35 30 36 36 21 24 18 10 43 28 15 26 27 14 29 19 29 31 41 20 44 42 26 19 16 39 28 21 39 29 20 21 24 17 13 15 15 16 28'.split(" "))
# breaks = [float(a) for a in breaks]
# wool = 'A A A A A A A A A A A A A A A A A A A A A A A A A A A B B B B B B B B B B B B B B B B B B B B B B B B B B B'.split(' ')
# tension = 'L L L L L L L L L M M M M M M M M M H H H H H H H H H L L L L L L L L L M M M M M M M M M H H H H H H H H H'.split(' ')
# ##############
# #response variable y must be the leftmost object in the dataframe, the first entered
# #creating an lm object, 2 steps
#  # 1.) make a dataframe (df)
#  # 2.) lm_object: lm('y ~ x*', data = df) == ols('y ~ x*', data = df).fit()

# #data MUST be entered with response being first in lm and dataframe (on the leftmost)
# #it should only have a format with the y and x's being in their place below
# # ols(y ~ ax + bx + cx + ... + x*, data = df).fit()
# warpbreaks = pd.DataFrame({'breaks':breaks,'wool':wool,'tension':tension})
# lmout = ols('breaks ~ wool + tension',warpbreaks).fit()
# print(anovatolint(lmout, data = warpbreaks, alpha = 0.10, P = 0.95, side = 2, method = "HE"))
