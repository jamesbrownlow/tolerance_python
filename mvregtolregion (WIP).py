import scipy.stats as st
import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
import patsy
import inspect

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
    


def mvregtolregion(mvreg, formI, df, names, newx = None, alpha = 0.05, P = 0.99, B = 1000):
    try:
        mvreg[0].params
    except:
        return "mvreg must be an lm object"
    else:
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
        for i in range(length(mvreg)):
            yhat.append(mvreg[i].predict())
        yhat = pd.DataFrame(np.array(yhat).T,columns = yvars)
        if type(newx) != None:
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
        A = np.linalg.inv(np.dot(np.dot(x.T,Pn),x))
        d2 = []
        # xall.iloc[0] == x.all[1,]
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
            L.append(np.array(np.linalg.eigvals(rwishart(fm,q))).T)
            for j in range(length(L[0])):
                if j == 0:
                    tmpc1 = []
                    tmpc2 = [] 
                    tmpc3 = []
                tmpc1.append((1+(1 * H2[i][j]**2)/(L[i][j]**1)))
                tmpc2.append((1+(2 * H2[i][j]**2)/(L[i][j]**2)))
                tmpc3.append((1+(3 * H2[i][j]**2)/(L[i][j]**3)))
                if j == length(L[0])-1:
                    c1.append(sum(tmpc1))
                    c2.append(sum(tmpc2))
                    c3.append(sum(tmpc3))
        # tt = []
        # L = np.expand_dims(L,axis=-1)
        # for i in range(B):
        #     tt.append(list(map(sum,(1+(1 * H2[i].T**2)/(L[i].T**1)))))
        # print(tt)
        a = np.array(c2)**3/np.array(c3)**2
        Tall = []
        a = pd.DataFrame(a)
        c1 = pd.DataFrame(c1)
        c2 = pd.DataFrame(c2)
        c3 = pd.DataFrame(c3)
        for i in range(N):
            Tall.append((fm*(np.sqrt(c2.iloc[:,i])/a.iloc[:,i]) * (st.chi2.ppf(P, a.iloc[:,i]) - a.iloc[:,i]) + c1.iloc[:,i]))
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
        return tol





## Must be in order
grain = pd.DataFrame([40, 17, 9, 15, 6, 12, 5, 9])
straw = pd.DataFrame([53, 19, 10, 29, 13, 27, 19, 30])
fert = pd.DataFrame([24, 11, 5, 12, 7, 14, 11, 18])
df = pd.concat([grain,straw,fert],axis = 1)

# y's on left, x on right
df.columns = ('grain','straw','fert')
newx = pd.DataFrame({'fert':[10,15,20]})

#note this only works with 2 and only 2 y's
dfy = pd.concat([df.iloc[:,0],df.iloc[:,1]],axis=1)
# 1.) df = pandas.DataFrame({'response':response, 'x1':x1, 'x2':x2,
#                                        ...}))
# 2.) ols('response ~ x1 + x2 +...', data = df).fit()
lm = []
modelmatrix = []
formulaI = 'fert + I(fert**2)'

# 'fert + I(fert**2) changes, this is your right side of the equation (x)
for i in range(length(dfy.iloc[0])):
    formula = 'fert + I(fert**2)'
    formula = 'dfy.iloc[:,i] ~ ' + formula
    lm.append(ols(formula,data=df).fit())

#print(lm[0].predict(newx))
#print(lm[1].predict(newx))


#lm1 = ols(formula, data = df).fit()
#lm2 = ols(formula, data = df).fit()
#y's on left, x on right
names = ['grain','straw','fert']

print("Warning:\n kfactor needs to be fixed.\n")
print(mvregtolregion(lm, formulaI,df, names,newx,alpha = 0.05))




















