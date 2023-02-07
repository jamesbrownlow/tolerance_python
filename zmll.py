from scipy.optimize import minimize
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
import pandas as pd
import sympy as sym
from scipy.misc import derivative
import scipy.optimize as opt

def length(x):
    if type(x) == float or type(x) == int or type(x) == np.int32 or type(x) == np.float64 or type(x) == np.float32 or type(x) == np.int64:
        return 1
    return len(x)

def zetafun(x):
    '''
Zeta Function

Description
    Zeta function, internal

Usage
    zetafun(x)
    
Parameters
----------
    x:
        For zetafun, a vector or matrix whose real values must be greater than 
        or equal to 1.

Details
-------
    This functions are not intended to be called by the user. zetafun is a 
    condensed version of the Riemann's zeta function given in R's VGAM package.
    Please use that reference if looking to directly implement Riemann's zeta 
    function. The function we have included is done so out of convenience.

References
----------
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.

    Yee, T. (2010), The VGAM Package for Categorical Data Analysis, Journal of 
        Statistical Software, 32, 1–34.
    
Example
-------
    zetafun([2,3,4,5,6])
    '''
    x = np.array(x)
    if any(x < 1): 
        return "Invalid input for Riemann's zeta function."
    a = 12
    k = 8
    B = [1/6, -1/30, 1/42, -1/30, 5/66, -691/2730, 7/6, -3617/510]
    ans = []
    for i in range(1,a):
        ans.append(1/i**x[i%length(x)-1])
    ans = np.array(ans)
    ans = np.sum(ans)
    ans = ans + 1/((x - 1) * a**(x - 1)) + 1/(2 * a**x)
    term = (x/2)/a**(x + 1)
    ans = ans + term * B[0]
    for mm in range(1,k):
        term = term * (x + 2 * mm - 2) * (x + 2 * mm - 3)/(a * a * 2 * mm * (2 * mm - 1))
        ans = ans + term * B[mm]
    return ans

def zmll(x, N = None, s = 1, b = 1, dist = 'Zipf'):
    '''
Maximum Likelihood Estimation for Zipf-Mandelbrot Models

Description
    Performs maximum likelihood estimation for the parameters of the Zipf, 
    Zipf-Mandelbrot, and zeta distributions.

Usage
    zmll(x, N = None, s = 1, b = 1, dist = ["Zipf", "Zipf-Man", "Zeta"]) 
    
Parameters
----------
    x:	
        A vector of raw data or a table of counts which is distributed 
        according to a Zipf, Zipf-Mandelbrot, or zeta distribution. Do not 
        supply a vector of counts!

    N:
        The number of categories when dist = "Zipf" or dist = "Zipf-Man". This
        is not used when dist = "Zeta". If N = None, then N is estimated based
        on the number of categories observed in the data.

    s:
        The initial value to estimate the shape parameter, which is set to 1 
        by default. If a poor initial value is specified, then a warning 
        message is returned.

    b:	
        The initial value to estimate the second shape parameter when 
        dist = "Zipf-Man", which is set to 1 by default. If a poor initial 
        value is specified, then a warning message is returned.

    dist:
        Options are dist = "Zipf", dist = "Zipf-Man", or dist = "Zeta" if the 
        data is distributed according to the Zipf, Zipf-Mandelbrot, or zeta 
        distribution, respectively.

Details
    Zipf-Mandelbrot models are commonly used to model phenomena where the 
    frequencies of categorical data are approximately inversely proportional 
    to its rank in the frequency table.

Returns
-------
    zmll returns a dataframe with coefficients

Note
    This function may be updated in a future version of the package so as to 
    allow greater flexibility with the inputs.

References
    Derek S. Young (2010). tolerance: An R Package for Estimating Tolerance 
        Intervals. Journal of Statistical Software, 36(5), 1-39. 
        URL http://www.jstatsoft.org/v36/i05/.
        
    Mandelbrot, B. B. (1965), Information Theory and Psycholinguistics. In B. 
        B. Wolman and E. Nagel, editors. Scientific Psychology, Basic Books.

    Zipf, G. K. (1949), Human Behavior and the Principle of Least Effort, 
        Hafner.

    Zornig, P. and Altmann, G. (1995), Unified Representation of Zipf 
        Distributions, Computational Statistics and Data Analysis, 19, 461–473.

Examples
    ## Maximum likelihood estimation for randomly generated data
    ## from the Zipf, Zipf-Mandelbrot, and zeta distributions. 
        N = 30
        
        s = 2
        
        b = 5
        
        Zdata = [6, 2, 1, 4, 8, 3, 3, 14, 2, 1, 21, 5, 18, 2, 30, 10, 8, 2, 
                 11, 4, 16, 13, 17, 1, 7, 1, 1, 28, 19, 27, 2, 7, 7, 13, 1,
                 15, 1, 16, 9, 9, 7, 29, 3, 10, 3, 1, 20, 8, 12, 6, 11, 5, 1,
                 5, 23, 3, 3, 14, 6, 9, 1, 24, 5, 11, 15, 1, 5, 5, 4, 10, 1,
                 12, 1, 3, 4, 2, 9, 2, 1, 25, 6, 8, 2, 1, 1, 1, 4, 6, 7, 26, 
                 10, 2, 1, 2, 17, 4, 3, 22, 8, 2]
        
    ## Zipf
        zmll(x = Zdata, N = N, s = s, b = b, dist = 'Zipf')
    
    ## Zipf-Mandelbrot
        zmll(x = Zdata, N = N, s = s, b = b, dist = 'Zipf-Man')
    
    # Zeta
        zmll(x = Zdata, N = np.inf, s = s, b = b, dist = 'Zeta')
    '''
    x = pd.DataFrame(x)
    x = pd.DataFrame(x.value_counts()).T
    x.columns = list(range(0,length(x.iloc[0])))
    Ntemp = length(x.iloc[0])
    x = x.reindex(np.argsort(x.columns),axis=1)
    if dist == 'Zeta':
        N = Ntemp
    if N == None:
        N = Ntemp
    if N < Ntemp:
        return "N cannot be smaller than the maximun number of categories in x!"
    Nseq = np.array(list(range(1,N+1)))
    zeros = np.zeros(N-length(x.iloc[0]))
    zeros = [int(z) for z in zeros]
    zeros = pd.DataFrame(zeros).T
    zeros.columns = ['']*length(zeros.columns)
    x.iloc[0] = x.iloc[0]
    x = pd.concat([x,zeros],axis=1)
    x = x.iloc[0].to_numpy()
    if dist == 'Zipf':
        def llzipf(s):
            return sum(x*(s*np.log(Nseq)+np.log(sum(1/(Nseq)**s))))
        fit = opt.minimize(llzipf, x0=0, method = 'BFGS')['x']
        fit = pd.DataFrame(fit,columns = ['s'])
        fit.index = ['Coefficients']
    if dist == "Zipf-Man":
        def llzima(params = [s,b]):
            return sum(x*(params[0]*np.log(Nseq+params[1])+np.log(sum(1/(Nseq+params[1])**params[0]))))
        fit = opt.minimize(llzima, x0=[0,0],method = 'L-BFGS-B')['x']
        fit = pd.DataFrame(fit).T
        fit.index = ['Coefficients']
        fit.columns = ['s','b']
    if dist == "Zeta":
        def llzeta(s):
            return sum(x*(s*np.log(Nseq)+np.log(zetafun(s))))
        fit = opt.minimize(llzeta, x0=1+1e-14, method = 'BFGS')['x']
        fit = pd.DataFrame(fit,columns = ['s'])
        fit.index = ['Coefficients']
    return fit
        