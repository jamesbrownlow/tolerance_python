import scipy.optimize
import numpy as np
import pandas as pd

def length(x):
    if type(x) == float or type(x) == int:
        return 1
    return len(x)

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
            ret = int(np.ceil(scipy.optimize.brentq(f,a=0,b=1e100,args=(alpha,P),maxiter=1000)))
        if P == None:
            ret = np.ceil(bisection(0,1,alpha =alpha, n = n, tol = 1e-8)*10000)/10000    
    return ret

def distfreeest(n = None, alpha = None, P = None, side = 1):
    '''
Estimating Various Quantities for Distribution-Free Tolerance Intervals

Description
    When providing two of the three quantities n, alpha, and P, this function 
    solves for the third quantity in the context of distribution-free 
    tolerance intervals.

Usage
    distfreeest(n = None, alpha = None, P = NULL, side = 1)

Parameters
----------
    n : int or list of ints, optional
        The necessary sample size to cover a proportion P of the population 
        with confidence 1-alpha. Can be a vector. The default is None.
        
    alpha : float or list of floats, optional
        1 minus the confidence level attained when it is desired to cover a 
        proportion P of the population and a sample size n is provided. Can be
        a vector. The default is None.
        
    P : float or list of floats, optional
        The proportion of the population to be covered with confidence 1-alpha 
        when a sample size n is provided. Can be a vector. The default is None.
        
    side : 1 or 2, optional
        Whether a 1-sided or 2-sided tolerance interval is assumed 
        (determined by side = 1 or side = 2, respectively). The default is 1.

Returns
-------
    out : dataframe
        When providing two of the three quantities n, alpha, and P, 
        distfreeest returns the third quantity. If more than one value of a 
        certain quantity is specified, then a table will be returned.
        
References
    Natrella, M. G. (1963), Experimental Statistics: National Bureau of 
        Standards - Handbook No. 91, United States Government Printing Office,
        Washington, D.C.
    
Examples
    # Solving for 1 minus the confidence level.

        distfreeest(n = 59, P = 0.95, side = 1)
    
    ## Solving for the sample size.
    
        distfreeest(alpha = 0.05, P = 0.95, side = 1)
    
    ## Solving for the proportion of the population to cover.
    
        distfreeest(n = 59, alpha = 0.05, side = 1)
    
    ## Solving for sample sizes for many tolerance specifications.
    
        distfree.est((alpha = [0.01,0.02,0.05], P = [0.95,0.99],side = 2)

    '''
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
        #print(f'length of alpha = {A}',f'length of n = {B}')
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

# print(distfreeest(n = 59, P = 0.95, side = 1))
# print(distfreeest(alpha = 0.05, P = 0.95, side = 1))
# print(distfreeest(n = 59, alpha = 0.05, side = 1))
#print(distfreeest(alpha = [0.01,0.02,0.05], P = [0.95,0.99],side = 2))