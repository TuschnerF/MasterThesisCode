# This is an implementation of the reconstruction operator of the 
# circular mean Radon transform used in PAT or TAT

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.special import hyp2f1
import math
import time
from rich.progress import Progress
from multiprocessing import Pool
from tqdm import tqdm
from joblib import Parallel, delayed

def imaging_operator(image, p, q) -> NDArray:
    '''
    Implementation of the Radon transform of an 2d image

    Parameters:
        image: numpy.ndarray 
            2D image
        p: int
            number of different angles
        q: int
            number of different radii

    Returns:
    radon : numpy.ndarray (2q+1,p)
        radondata for angles and radii
    '''
    angles = np.linspace(0,np.pi, p, endpoint=False)
    radii = np.linspace(1,-1,2*q+1, endpoint=True)
    radon = np.zeros((len(angles), len(radii)))
    rows, columns = image.shape
    if rows != columns:
        raise ValueError("Image not quadratic")
    
    n_points_line = 1000
    for i, theta in enumerate(angles):
        for j, s in enumerate(radii):
            Ts = np.sqrt(1-s*s)
            r = np.linspace(-Ts, Ts, n_points_line, endpoint=False)
            x_circle = (1 + s*np.cos(theta) - r*np.sin(theta)) * columns / 2
            y_circle = (1 - s*np.sin(theta) - r*np.cos(theta)) * rows / 2
            
            # bilinear interpolation of the x,y values in the image
            sampled_values = map_coordinates(image, [y_circle, x_circle], order=1, mode='constant')
            
            # Compute the average value over the sampled points (arc length measure)    	
            tmp = 2*Ts
            radon[i, j] = tmp*np.mean(sampled_values)
            if np.isnan(radon[i, j]):
                radon[i, j] = 0
    return radon

def draw_rectangle(ur, vr, phi, cx, cy, p):
    """
    Characteristic function of a rectangle in [-1, 1]^2.

    Parameters:
    ur, vr : float
        Half side lengths of the rectangle.
    phi : float
        Rotation angle in radians.
    cx, cy : float
        Center coordinates of the rectangle.
    p : int
        Discretization parameter.

    Returns:
    rect : numpy.ndarray
        A 2D binary array representing the digitized characteristic function
        of the rectangle with shape (2*p+1, 2*p+1).
    """
    dim = 2 * p + 1
    h = 1 / p

    rect = np.zeros((dim, dim))

    # Rotation matrix components
    c = np.cos(phi)
    s = np.sin(phi)

    for i in range(0, 2 * p + 1):
        x = -1 + i * h - cx  # x-coordinate in the grid
        for j in range(0, 2 * p + 1):
            y = -1 + j * h - cy  # y-coordinate in the grid

            # Transform coordinates to rectangle-aligned system
            z1 = (c * x + s * y) / ur
            z2 = (-s * x + c * y) / vr

            # Check if the point lies within the rectangle
            if abs(z1) <= 1 and abs(z2) <= 1:
                rect[dim - j - 1, i] = 1  # Flip y-axis to match image convention

    return rect

def draw_ellipse(a, b, cx, cy, p):
    """
    Characteristic function of a rectangle in [-1, 1]^2.

    Parameters:
    a,b : floar
        ellipse parameter
    p : int
        Discretization parameter.

    Returns:
    rect : numpy.ndarray
        A 2D binary array representing the digitized characteristic function
        of the rectangle with shape (2*p+1, 2*p+1).
    """
    dim = 2 * p + 1
    h = 1 / p
    ellipse = np.zeros((dim, dim))

    for i in range(0, 2 * p + 1):
        x = -1 + i * h - cx  # x-coordinate in the grid
        for j in range(0, 2 * p + 1):
            y = -1 + j * h + cy  # y-coordinate in the grid
            if (x*x/(a*a) + y*y/(b*b)) <= 1:
                ellipse[j,i] = 1        
    return ellipse

def projrec(ur,vr,phi,cx,cy,q,p):
    """
    Radon-Transformation of a rectangle.

    Parameters:
    ur, vr : float
        Half side lengths of the rectangle.
    phi : float
        Rotation angle in radians.
    cx, cy : float
        Center coordinates of the rectangle.
    q : int
        Discretization parameter in s-direction
    p : int
        Discretization parameter in angle.

    Returns:
    sinogram : numpy.ndarray
        Sinogramm of indicator function for rectamgle
        shape [p 2*q+1], integal(j,i)=integral along line ( (i-q)/q,j*pi/p)
    """

    print("S0: Calculate Radon transform (sinogramm)")
    dim = 2*q+1
    h=1/q
    delta=np.pi/p

    sinogram = np.zeros((p, dim))

    for j in range(p):
        angle = -phi + delta * j

        cosi = np.cos(angle)
        sinu = np.sin(angle)

        ca = -sinu / ur
        cb = cosi / vr

        atp = np.sqrt(ca**2 + cb**2)
        shift = cx * np.cos(angle + phi) + cy * np.sin(angle + phi)
        gamma = np.arctan2(cb, ca) - np.pi / 2

        if gamma < 0:
            gamma += 2 * np.pi

        for i in range(dim):
            sprime = (i - q) * h + shift
            u = sprime * (cosi * np.cos(gamma) / ur + sinu * np.sin(gamma) / vr)

            if gamma >= np.pi:
                gamma -= np.pi
                u = -u

            cog = abs(np.cos(gamma))
            sig = abs(np.sin(gamma))

            if cog > 1e-6 and sig > 1e-6:
                tmp = min(cog, u + sig) - max(-cog, u - sig)
                if tmp > 0:
                    sinogram[j, i] = tmp / (cog * sig * atp)
            else:
                if abs(u) < 1:
                    sinogram[j, i] = 2 / atp

    return sinogram

def projellipse(a, b, cx, cy, q, p):
    """
    Radon transform of an ellipse.

    Parameters:
    a,b: floor
        ellipse parameters
    q : int
        Discretization parameter in s-direction
    cx, cy : float
        Center coordinates of the rectangle.
    p : int
        Discretization parameter in angle.

    Returns:
    sinogram : numpy.ndarray
        Sinogramm of indicator function for rectamgle
        shape [p 2*q+1], integal(j,i)=integral along line ( (i-q)/q,j*pi/p)
    """
    dim = 2*q+1
    sinogram = np.zeros((p, dim))
    for j in range(p):
        theta = j*np.pi/p
        for i in range(dim):
            r = np.sqrt((a*np.cos(theta))**2 + (b*np.sin(theta))**2)
            s = (i-q)/q + (cx*np.cos(theta)+cy*np.sin(theta))
            if np.abs(s) <= r: 
                sinogram[j,i] = (2*a*b/r) * (np.sqrt(1-s*s/(r*r))) 
    return sinogram

def kernel_4_11(s_values,n):
    """
    Calculate Kernel v^n(s). Exmple 4.11 in the lecture script "Mathematical Methods in Imaging" by A. Rieder 2014

    Parameters:
    s_values : np.ndarray
    n : int
        regularization parameter, n = 1/gamma

    Returns:
    res : nd.array 
    """
    res = np.zeros_like(s_values)
    tmp = 2*np.pi*np.pi
    for i, s in enumerate(s_values):
        if np.abs(s) <= 1:
            res[i] = 2*(n+1)*hyp2f1(1,-n,1/2, s**2)
        else:
            res[i] = -hyp2f1(1, 3/2, n+2, 1/(s*s))/(s*s)
    return res / tmp

def shepp_logan_filter(s_values,n):
    """
    Calculate Kernel for Shepp logan filter

    Parameters:
    s_values : np.ndarray
    n : int
        regularization parameter, n = 1/gamma    NOT NEEDED

    Returns:
    res : nd.array 
    """
    res = np.zeros_like(s_values)
    q = 0
    if (len(s_values)-1) % 4 == 0:
        q = (len(s_values)-1) // 4
    else:
        print("Wrong dimension in shepp-logan-filter")
    center = 2*q
    res[center]=2*(q*np.pi)**2
    for x in range(2*q+1):
        res[center+x] = res[center] / (1-4*(x+1)*(x+1))
        res[center-x] = res[center+x]
    return res *(-1)
    
def draw_roi(orig, lr, value = 0.5):
    """
    adds a circle in to a picture to show the ROI

    Parameters:
    orig : numpy.ndarray
        original picture
    lr : float
        limited radius
    value : float
        sets value of the circle-pixels

    Returns:
    orig : numpy.ndarray
        changed original picture
    """
    p,q = orig.shape
    for i in range(0, p):
        x = -1 + i * (2 / p)   # x-coordinate in the grid
        for j in range(0, p):
            y = -1 + j * (2 / p)  # y-coordinate in the grid
            if (np.sqrt(x*x + y*y) <= lr+0.005) and (np.sqrt(x*x + y*y) >= lr-0.005):
                orig[j,i] = value
    return orig    

def filter(sinogramm, q, p, n, lr = 1.0, la = 0.0,  cutoff = False):
    """
    calculates the convolution of the kernel and sinogram, while considering limited data 

    Parameters:
    sinogram : numpy.ndarray (2q+1,p)
        sinogram
    q : float
        number of angles
    p : float
        number of radii
    n : float
        regularizing parameter for calculating the kernel
    lr : float
        limited radius
    la : float
        limited angle
    cutoff : boolean
        describes if smooth cut-off dunction is used

    Returns:
    eta : numpy.ndarray
        filtered data
    """
    # S0: Limited angle / radius
    if lr>=0.0 and lr<1.0:
        print("Limited radius with |r|<=", lr)
        for j in range(2*q+1):
            if abs((j-q)/q)>=lr:
                sinogramm[:,j] = 0
        if cutoff == True:
            eps = lr/5
            for j in range(2*q+1):
                r = abs((j-q)/q)
                if r>=lr-eps and r<=lr:
                    # print("r = "+ r +"  cutoff = "+ cut_off_function(-lr+eps+np.abs(r),eps))
                    sinogramm[:,j] *= cut_off_function(-lr+eps+np.abs(r),eps)
    if la>0.0 and la<=np.pi:
        print("Limited angle with |alpha|<=", la)
        for j in range(p):
            if j*np.pi/p<= la or j*np.pi/p>=np.pi-la:
                sinogramm[j,:] = 0
            if cutoff == True:
                phi=j*np.pi/p
                eps = (np.pi-2*la)/3
                if la< phi and phi < la+eps:
                    sinogramm[:,j] *= cut_off_function(la+eps-phi, eps)
                if np.pi-la-eps < phi and phi < np.pi-la:
                    sinogramm[:,j] *= cut_off_function(phi+la-np.pi+eps, eps)
        
    # S1:   Calculate Filter
    print("S1: Calculation of reconstruction filter")
    eta = np.zeros_like(sinogramm)
    # v_gamma = kernel_4_11(np.linspace(-2*q,2*q,4*q+1)/q, n)
    v_gamma = shepp_logan_filter(np.linspace(-2*q,2*q,4*q+1)/q, n)
    for j in range(p):
       for k in range(2*q+1):
           for l in range(2*q+1):
               eta[j,k] += 1/q * ( v_gamma[k-l+2*q] * sinogramm[j,l] )
    return eta

def filtered_backprojection_paralell(eta, q, p, p_rec):
    """
    Algrorithm 4.16

    Parameters:
    sinogram : numpy.ndarray
        Sinogramm of indicator function for rectamgle
        shape [p 2*q+1], integal(j,i)=integral along line ( (i-q)/q,j*pi/p)
    
    p : int
        Discretization parameter in angle.
    q : int
        Discretization parameter in s-direction
    p_rec:  int
        Discretization parameter for reconstruction picture
    n : int
        regularization parameter
    lr : floor
        limited radius
    la : floor
        limited angle
    cutoff: bool
        if enabled use a smooth cutoff function

    Returns:
    res : numpy.ndarray
        reconstructed Picture
        shape [p_rec p_rec]
    """
    #Assert correct shape of sinogramm
    # S2:   Calculate reconstruction
    print("S2: Reconstruction for ", p_rec*p_rec, " points")
    res = np.zeros((p_rec,p_rec))
    for x1 in range(p_rec):
        for x2 in range(p_rec):
            x = [ 2*x1/(p_rec-1) - 1, 2*x2/(p_rec-1) - 1]
            if np.sqrt(x[0]**2+x[1]**2)<1: 
                for j in range(p):
                    # omega = [np.cos(j*np.pi/p),np.sin(j*np.pi/p)]
                    t = (2*x1/(p_rec-1) - 1)*np.sin(j*np.pi/p) + (2*x2/(p_rec-1) - 1)*-np.cos(j*np.pi/p)
                    t *= q
                    k = math.floor(t)
                    rho = t-k
                    res[x1,x2] += (np.pi / p) * ( (1-rho)*eta[j,k+q] + rho*eta[j,k+1+q])
    # norm to max value 1
    res *= 1 / (np.max(res))                
    return res

def filtered_backprojection_paralell_paralell(eta, q, p, p_rec):
    """
    Algrorithm 4.16 paralellized

    Parameters:
    sinogram : numpy.ndarray
        Sinogramm of indicator function for rectamgle
        shape [p 2*q+1], integal(j,i)=integral along line ( (i-q)/q,j*pi/p)
    
    p : int
        Discretization parameter in angle.
    q : int
        Discretization parameter in s-direction
    p_rec:  int
        Discretization parameter for reconstruction picture
    n : int
        regularization parameter
    lr : floor
        limited radius
    la : floor
        limited angle
    cutoff: bool
        if enabled use a smooth cutoff function

    Returns:
    res : numpy.ndarray
        reconstructed Picture
        shape [p_rec p_rec]
    """
    #Assert correct shape of sinogramm
    # S2:   Calculate reconstruction
    print("S2: Reconstruction for ", p_rec*p_rec, " points")
    res = np.zeros((p_rec,p_rec))
    
    def process_pixel(x1):
        row_result = np.zeros(p_rec)
        for x2 in range(p_rec):
            x = [ 2*x1/(p_rec-1) - 1, 2*x2/(p_rec-1) - 1]
            if np.sqrt(x[0]**2+x[1]**2)<1: 
                for j in range(p):
                    # omega = [np.cos(j*np.pi/p),np.sin(j*np.pi/p)]
                    t = (2*x1/(p_rec-1) - 1)*np.sin(j*np.pi/p) + (2*x2/(p_rec-1) - 1)*-np.cos(j*np.pi/p)
                    t *= q
                    k = math.floor(t)
                    rho = t-k
                    row_result[x2] += (np.pi / p) * ( (1-rho)*eta[j,k+q] + rho*eta[j,k+1+q])
        return row_result
    
    results = Parallel(n_jobs=-1)(delayed(process_pixel)(x1) for x1 in range(p_rec))
    for x1, row in enumerate(results):
        res[x1, :] = row
    
    # norm to max value 1
    res *= 1 / (np.max(res))                
    return res

def cut_off_function(x,eps):
    v_eps = np.exp(x**2/(x**2-eps**2))
    return v_eps
