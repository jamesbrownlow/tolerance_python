import scipy.stats as st
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import patsy

def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    return len(x)

def rwishart(df, p):
    '''
Random Wishart Distributed Matrices

Usage
    rwishart(df, p)
    

Parameters
----------
    df : int
        The degrees of freedom for the Wishart matrix to be generated.
    p : int
        The dimension of the random Wishart matrix.

Returns
-------
    X : matrix
        Random generation of Wishart matrices.
        
References
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Yee, T. (2010), The VGAM Package for Categorical Data Analysis, Journal of 
        Statistical Software, 32, 1–34.

Example:
    ## Generate a 4x4 wishart matrix with 10 degrees of freedom.
        
        rwishart(df = 10, p = 4)
    '''
    X = np.zeros(p*p)
    shape = (p,p)
    X = X.reshape(shape)
    #X = pd.DataFrame(X)
    #print(X.iloc[0][1])
    chi2rvs = []
    for i in range(p):
        chi2rvs.append(np.sqrt(st.chi2.rvs(size = 1,df = df-i)))
    np.fill_diagonal(X,chi2rvs)
    if p > 1:
        a = []
        for i in range(1,p):
            pseq = list(range(1,p))
            a.extend(np.repeat(4*pseq[i-1], i))
        # a is equivalent to rep(p*pseq,pseq)
        #X = pd.DataFrame(X)
        for i in range(p-1):
            for j in range((p-1)-i):
                X[i][j+1+i] = st.norm.rvs(size = 1)
        X = np.dot(X.T,X)
        return X
    
#https://www.geeksforgeeks.org/python-dividing-two-lists/
def divide_lists(list1, list2):
    if not list1 or not list2:
        return []
    return [list1[0] / list2[0]] + divide_lists(list1[1:], list2[1:])

def mvregtolregion(mvreg, formI, df, newx = None, alpha = 0.05, P = 0.99, B = 1000):
    '''
Multivariate (Multiple) Linear Regression Tolerance Regions

Description
    Determines the appropriate tolerance factor for computing multivariate 
    (multiple) linear regression tolerance regions based on Monte Carlo 
    simulation.

Usage
    mvregtolregion(mvreg, formI, df, newx = None, alpha = 0.05, P = 0.99, 
                   B = 1000)

Parameters
----------
    mvreg : multiple linear model
        A multivariate (multiple) linear regression fit, having class mlm.
        
    formI : string
        The formula of the right-hand side. y = x*.
        
    df : DataFrame
        The DataFrame that holds all data.
        
    newx : DataFrame, optional
        An optional data frame of new values for which to approximate 
        k-factors. This must be a data frame with named columns that match 
        those in the data frame used for the mvreg fitted object. This can 
        only be done for new x's, not new y's. The default is None.
        
    alpha : float, optional
        The level chosen such that 1-alpha is the confidence level. The 
        default is 0.05.
        
    P : float, optional
        The proportion of the population to be covered by this tolerance 
        region. The default is 0.99.
        
    B : int, optional
        The number of iterations used for the Monte Carlo algorithm which 
        determines the tolerance factor. The number of iterations should be at 
        least as large as the default value of 1000.

Returns
-------
    mvregtolregion returns a matrix where the first column is the k-factor, 
    the next q columns are the estimated responses from the least squares fit, 
    and the final m columns are the predictor values. The first n rows of the
    matrix pertain to the raw data as specified by y and x. If values for newx 
    are specified, then there are is one additional row appended to this output 
    for each row in the matrix newx.

Details
-------
    A basic sketch of how the algorithm works is as follows:
        1. Generate independent chi-square random variables and Wishart 
        random matrices.
        
        2. Compute the eigenvalues of the randomly generated Wishart matrices.
        
        3. Iterate the above steps to generate a set of B sample values such 
        that the 100(1-alpha)-th percentile is an approximate tolerance factor.
    
References
----------

    Anderson, T. W. (2003) An Introduction to Multivariate Statistical 
        Analysis, Third Edition, Wiley.

    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.

    Krishnamoorthy, K. and Mathew, T. (2009), Statistical Tolerance Regions: 
        Theory, Applications, and Computation, Wiley.

    Krishnamoorthy, K. and Mondal, S. (2008), Tolerance Factors in Multiple 
        and Multivariate Linear Regressions, Communications in Statistics - 
        Simulation and Computation, 37, 546–559.

Examples
--------
    grain = pd.DataFrame([40, 17, 9, 15, 6, 12, 5, 9],columns = ['grain'])
    
    straw = pd.DataFrame([53, 19, 10, 29, 13, 27, 19, 30], columns = ['straw'])  
    
    flower = pd.DataFrame([90,21,45,67,23,28,60,99],columns = ['flower'])
    
    fert = pd.DataFrame([24, 11, 5, 12, 7, 14, 11, 18],columns = ['fert'])
    
    ## Must be in order, y's on left, x on right
    
    df = pd.concat([grain,straw,flower,fert],axis = 1)
    
    #optional argument
    
    newx = pd.DataFrame({'fert':[10,15,20]})
    
    #######no need to touch
    
    dfy = []
    
    for i in range(length(df.iloc[0]-1)):
        dfy.append(df.iloc[:,i])
        
    dfy = dfy[:length(dfy)-1]
    
    dfy = pd.DataFrame(dfy).T
    ##############
    
    #your right side (x) of the equation, this must be changed to fit your equation
    
    formulaI = 'fert + I(fert**2)'
    
    #######no need to touch
    
    lm = []
    
    for i in range(length(dfy.iloc[0])):
        formula = formulaI
        
        formula = 'dfy.iloc[:,i] ~ ' + formula
        
        lm.append(ols(formula,data=df).fit())
    ##############
    
    mvregtolregion(lm, formulaI, df, newx, alpha = 0.01, P = 0.95, B = 5000)
    '''
    try:
        mvreg[0].params
    except:
        return "mvreg must be a mlm object"
    else:
        if length(df.iloc[0]) == 2:
            return 'mvreg must be a mlm object'
        names = df.columns
        X = pd.DataFrame(patsy.dmatrix(formI))
        xformulanames = formI.split("+")
        n = length(X.iloc[:,0])
        q = length(mvreg)
        yvars = names[:-1:1]
        if all(X.iloc[:,0] == 1):
            m = length(X.iloc[0]) -1
            loc = []
            for i in range(length(X.iloc[0]) -1):
                loc.append(i+1)
            x = X.loc[:,loc]
            xvars = x.columns
        else:
            m = length(X.iloc[0])
            x = X
            xvars = X.columns
        xbar = []
        for i in range(length(X.iloc[0])-1):
            xbar.append(np.mean(X.iloc[:,i+1]))
        fm = n-m-1
        Pn = np.zeros(shape=(n,n)) + 1
        Pa = np.zeros(shape=(n,n))
        np.fill_diagonal(Pa,1)
        Pn = Pa - (Pn/n)
        xvars = x.columns
        yhat = []
        for i in range(q):
            yhat.append(mvreg[i].predict())
        yhat = pd.DataFrame(np.array(yhat).T,columns = yvars)
        if type(newx) == pd.core.frame.DataFrame:
            nn = length(newx.iloc[:,0])
            newxy = np.array(range(1,q*nn+1))
            newxy = pd.DataFrame(newxy.reshape([q,nn]).T)
            newxy = pd.concat([newxy, newx],axis=1)
            newxy.columns = names
            yhatn = []
            for i in range(length(yvars)):
               yhatn.append(mvreg[i].predict(newx))
            yhatn = pd.DataFrame(yhatn).T
            yhatn.columns = yvars
            yhat = pd.concat([yhat,yhatn])
            yhat.index = range(length(yhat.iloc[:,0]))
            newx = pd.DataFrame(patsy.dmatrix(formI,data=newxy)).iloc[:, 1:]
        xall = pd.concat([x,newx])
        xall.columns = xformulanames
        xall.index = range(length(xall.iloc[:,0]))
        N = length(xall.iloc[:,0])
        A = np.linalg.inv(np.linalg.multi_dot([x.T,Pn,x]))
        d2 = []
        for i in range(N):
            d2.append((1/n)+np.linalg.multi_dot([xall.iloc[i]-xbar,A,xall.iloc[i]-xbar]))
        H2 = []
        L = []
        c1 = []
        c2 = [] 
        c3 = []
        for i in range(B): 
            H2.append(np.array(st.chi2.rvs(size = int(N*q),df=1)).reshape([q,N]))
            H2[i] = np.array([t*d2 for t in H2[i]])
        for i in range(B):
            L.append(np.array(sorted(np.linalg.eigvals(rwishart(fm,q)),reverse=True)).T)
        L = np.array(L)
        for i in range(B):
            c1.append(sum((1+H2[i]**2)/L[:,0][i]))
            c2.append(sum((1+2*H2[i]**2)/L[:,0][i]**2))
            c3.append(sum((1+3*H2[i]**2)/L[:,0][i]**3))
        a = np.array(c2)**3/np.array(c3)**2
        c1 = np.array(c1)
        c2 = np.array(c2)
        Tall = []
        for i in range(N):
            Tall.append((fm * ((np.sqrt(c2[:,i])/a[:,i]) * (st.chi2.ppf(P, a[:,i]) - a[:,i]) + c1[:,i])))
        k = list(map(lambda p: np.quantile(p,1-alpha),Tall))
        k = pd.DataFrame(k)
        tol = pd.concat([k,yhat,xall],axis =1)
        names = [n + '.hat' for n in names[0:-1]]
        formI = formI.split("+")
        returncol = []
        returncol.append('kfactor')
        returncol.extend(names)
        returncol.extend(formI)
        tol.columns = returncol
        print(f"These are the {(1-alpha)*100}%/{P*100}% tolerance factors")
        if type(newx) == pd.core.frame.DataFrame:
            indexnames = [list(range(length(X))),list(range(length(newxy)))]
            indexnames = [item for sublist in indexnames for item in sublist]
            indexnames[:length(X)] = ['X'+str(i+1) for i in indexnames[:length(X)]]
            indexnames[length(X):] = ['X'+str(i+1)+'.1' for i in indexnames[length(X):]]
            #print(indexnames)
        else:
            indexnames = range(length(X))
            indexnames = ['X'+str(i+1) for i in indexnames]
        tol.index = indexnames#range(length(X)),range(length(newxy))
        return tol

# example:
    
# grain = pd.DataFrame([40, 17, 9, 15, 6, 12, 5, 9],columns = ['grain'])
# straw = pd.DataFrame([53, 19, 10, 29, 13, 27, 19, 30], columns = ['straw'])
# flower = pd.DataFrame([90,21,45,67,23,28,60,99],columns = ['flower'])
# fert = pd.DataFrame([24, 11, 5, 12, 7, 14, 11, 18],columns = ['fert'])
# ## Must be in order, y's on left, x on right
# df = pd.concat([grain,straw,flower,fert],axis = 1)

# #optional argument
# newx = pd.DataFrame({'fert':[10,15,20]})

# #######no need to touch
# dfy = []
# for i in range(length(df.iloc[0]-1)):
#     dfy.append(df.iloc[:,i])
# dfy = dfy[:length(dfy)-1]
# dfy = pd.DataFrame(dfy).T
# ##############

# #your right side (x) of the equation, this must be changed to fit your equation
# formulaI = 'fert + I(fert**2)'

# #######no need to touch
# lm = []
# for i in range(length(dfy.iloc[0])):
#     formula = formulaI
#     formula = 'dfy.iloc[:,i] ~ ' + formula
#     lm.append(ols(formula,data=df).fit())
# ##############

# print(mvregtolregion(lm, formulaI, df, newx, alpha = 0.01, P = 0.95, B = 5000))










