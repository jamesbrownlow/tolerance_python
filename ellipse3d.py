import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Similar to ellipse3d() in R. 
# Need this function to plot the 3d tolerance region for mvtolregion
def ellipse3d(x, centre = np.zeros((3)), t = 3, smoothness = 1):
    """
Return the 3d points representing the covariance matrix x centred at centre 
    (Mean) and scaled by the factor t.

Usage
    get_cov_ellipsoid(x, centre = np.zeros((3)), t = 3)

Parameters
----------
    x: covariance matrix
        An object. In the default method the parameter x should be a square 
        positive definite matrix at least 3x3 in size. It will be treated as 
        the correlation or covariance of a multivariate normal distribution.
    
    centre: vector of length 3, optional
        The centre of the ellipse will be at this position.
        
    t: float, optional
        The size of the ellipse may also be controlled by specifying the value 
        of a t-statistic on its boundary. This defaults to the appropriate 
        value for the confidence region.
        
    smoothness: float, optional
        This controls the number of subdivisions used in constructing the 
        ellipsoid. Higher numbers give a smoother shape.

Returns
-------
    ellipse3d returns X, Y, and Z parameters intended to be used
    on 3d plots such as ax.plot_wireframe() and ax.plot_surface()

References
 https://github.com/CircusMonkey/covariance-ellipsoid/blob/master/ellipsoid.py

Examples
--------
    Example 1:  ax.plot_wireframe(X,Y,Z,alpha=0.2)
    Example 2:  ax.plot_surface(X,Y,Z,alpha=0.2)
    """
    assert x.shape==(3,3)
    # Find and sort eigenvalues to correspond to the covariance matrix
    eigvals, eigvecs = np.linalg.eigh(x)
    idx = np.sum(x,axis=0).argsort()
    eigvals_temp = eigvals[idx]
    idx = eigvals_temp.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    # Set of all spherical angles to draw our ellipsoid
    n_points = int(np.round(100*smoothness))
    theta = np.linspace(0, 2*np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)

    # Width, height and depth of ellipsoid
    rx, ry, rz = t * np.sqrt(eigvals)

    # Get the xyz points for plotting
    # Cartesian coordinates that correspond to the spherical angles:
    X = rx * np.outer(np.cos(theta), np.sin(phi))
    Y = ry * np.outer(np.sin(theta), np.sin(phi))
    Z = rz * np.outer(np.ones_like(theta), np.cos(phi))

    # Rotate ellipsoid for off axis alignment
    old_shape = X.shape
    # Flatten to vectorise rotation
    X,Y,Z = X.flatten(), Y.flatten(), Z.flatten()
    X,Y,Z = np.matmul(eigvecs, np.array([X,Y,Z]))
    X,Y,Z = X.reshape(old_shape), Y.reshape(old_shape), Z.reshape(old_shape)
   
    # Add in offsets for the mean
    X = X + centre[0]
    Y = Y + centre[1]
    Z = Z + centre[2]
    return X,Y,Z

    ## Example
# fig = plt.figure()
# ax = fig.add_subplot(111,projection = '3d')
# xdata = [np.random.normal(size = 100,loc=0,scale = 0.2), np.random.normal(size = 100,loc=0,scale = 0.5), np.random.normal(size = 100,loc=5,scale = 1)]
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# ax.set_xlim([min(xdata[0]),max(xdata[0])])
# ax.set_ylim([min(xdata[1]),max(xdata[1])])
# ax.set_zlim([min(xdata[2]),max(xdata[2])])
# ax.scatter3D(xdata[0],xdata[1],xdata[2],c=xdata[2])
# xdata = pd.DataFrame(xdata)
# Mean = xdata.mean(axis=1)
# Sigma = np.cov(xdata)
# tolout = np.sqrt(7.383685)
# ax.set_title("3D Tolerance Region")
# X, Y, Z = get_cov_ellipsoid(x = Sigma, centre = Mean, t = tolout, smoothness = 0.9)
# ax.plot_surface(X,Y,Z,alpha=0.3)
