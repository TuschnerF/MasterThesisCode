# This is an implementation of the reconstruction operator of the 
# circular mean Radon transform used in PAT or TAT

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.special import hyp2f1
from  problems.problem import Problem
import math
import time

class Problem_LAP(Problem):
    # Constructor
    def __init__(self):
        pass
    
    def imaging_operator(self, image: NDArray, angles : NDArray, radii) -> NDArray:
        pass


def rectangle(ur, vr, phi, cx, cy, p):
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

    rect = np.zeros((dim, dim), dtype=int)

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

def projrec(ur,vr,phi,cx,cy,q,p):
    """
    Projection of a rectangle.

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

def kernel_4_11(s_values,n):
    """
    Calculate Kernel v^n(s). Exmple 4.11 in Scrit Rieder

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
            res[i] = tmp*2*(n+1)*hyp2f1(1,-n,1/2, s**2)
        else:
            res[i] = -tmp*hyp2f1(1, 3/2, n+2, 1/(s*s))/(s*s)
    return res

def filtered_backprojection_paralell(sinogramm, q, p, p_rec, n, lr = 1.0):
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

    Returns:
    res : numpy.ndarray
        reconstructed Picture
        shape [p_rec p_rec]
    """
    #Assert correct shape of sinogramm
    # S0: Limited angle / radius
    if lr>=0.0 and lr<=1:
        print("Limited radius with |r|<=", lr)
        for j in range(2*q+1):
            if abs((j-q)/q)>=lr:
                sinogramm[:,j] = 0

    # S1:   Calculate Filter
    print("S1: Calculation of reconstruction filter")
    start_filter = time.perf_counter()
    eta = np.zeros_like(sinogramm)
    v_gamma = kernel_4_11(np.linspace(-2*q,2*q,4*q+1)/q, n)
    for j in range(p):
       for k in range(2*q+1):
           for l in range(2*q+1):
               eta[j,k] += 1/q * ( v_gamma[k-l+2*q] * sinogramm[j,l] ) # Indizes überprüfen!
    end_filter = time.perf_counter()


    # S2:   Calculate reconstruction
    print("S2: Reconstruction for ", p_rec*p_rec, " points")
    start_reconstruction = time.perf_counter()
    res = np.zeros((p_rec,p_rec))
    k_max = -100000
    k_min = 100000
    for x1 in range(p_rec):
        for x2 in range(p_rec):
            x = [ 2*x1/(p_rec-1) - 1, 2*x2/(p_rec-1) - 1]
            if np.sqrt(x[0]**2+x[1]**2)<1: 
                for j in range(p):
                    # omega = [np.cos(j*np.pi/p),np.sin(j*np.pi/p)]
                    t = (2*x1/(p_rec-1) - 1)*np.sin(j*np.pi/p) + (2*x2/(p_rec-1) - 1)*-np.cos(j*np.pi/p)
                    t *= q
                    k = math.floor(t)
                    k_max = max(k,k_max)
                    k_min = min(k,k_min)
                    rho = t-k
                    res[x1,x2] += (np.pi / p) * ( (1-rho)*eta[j,k+q] + rho*eta[j,k+1+q])
    # norm to max value 1
    res *= 1 / (np.max(res))                
    print("Reconstruction finished")
    end_reconstruction = time.perf_counter()
    print("Elapsed Time:")
    print("Calculation Filter:", end_filter-start_filter)
    print("Calculation Reconstruction:", end_reconstruction-start_reconstruction)
    return res