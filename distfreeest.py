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
        print(f'length of alpha = {A}',f'length of n = {B}')
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


