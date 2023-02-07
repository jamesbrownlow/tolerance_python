import numpy as np

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

