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

def mvregtolregion(mvreg, formI, df, names, newx = None, alpha = 0.05, P = 0.99, B = 1000):
    try:
        mvreg[0].params
    except:
        return "mvreg must be an lm object"
    else:
        X = pd.DataFrame(patsy.dmatrix(formI))
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
        Pa = Pa - (Pn/n)
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
        names = [n + '.hat' for n in names[0:-1]]
        yhat.columns = names
        formI = formI.split("+")
        x.columns = formI
        return pd.concat([yhat,x],axis=1)
            
        
        
        







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

#lm1 = ols(formula, data = df).fit()
#lm2 = ols(formula, data = df).fit()
#y's on left, x on right
names = ['grain','straw','fert']

print("This file is a work in progress.\n")
print(mvregtolregion(lm, formulaI,df, names,newx))




















