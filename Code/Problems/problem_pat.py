# This is an implementation of the reconstruction operator of the 
# circular mean Radon transform used in PAT or TAT

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator
from rich.progress import Progress
from multiprocessing import Pool
from tqdm import tqdm
from joblib import Parallel, delayed

def imaging_operator(image, p, q) -> NDArray:
    # Implementation of the circular mean Radon transform
    # image: 2D image
    # angles: Array with all phi for points on the sphere: Theta(phi)
    # radii: Array of all r for all circles
    angles = np.linspace(0,2*np.pi, p, endpoint=False)
    radii = np.linspace(0,2,q, endpoint=False)
    radon = np.zeros((len(angles), len(radii)))
    rows, columns = image.shape
    if rows != columns:
        raise ValueError("Image not quadratic")
    
    n_points_circle = 360
    for i,phi in enumerate(angles):
        for j, r in enumerate(radii):
            psi = np.linspace(0, 2*np.pi, n_points_circle, endpoint=False)
            x_circle = (1 + np.cos(phi) + r*np.cos(psi)) * columns / 2
            y_circle = (1 - np.sin(phi) - r*np.sin(psi)) * rows / 2
            
            # bilinear interpolation of the x,y values in the image
            sampled_values = map_coordinates(image, [y_circle, x_circle], order=1, mode='constant')
            
            # Compute the average value over the sampled points (arc length measure)    	
            radon[i, j] = np.mean(sampled_values)
    radon = diff_operator_radial(radon)
    return radon

def diff_operator_radial(radon) -> NDArray:
    dr_radon = radon.copy()
    P,Q = np.shape(radon)
    h = 2/Q
    for q in range(1,Q-1):
        dr_radon[:,q] = (-radon[:,q-1]+2*radon[:,q] -radon[:,q+1]) / (h*h)
    # q=0
    dr_radon[:,0] = (radon[:,2]-2*radon[:,1] - radon[:,0]) / (h*h)
    # q=Q
    dr_radon[:,Q-1] = (radon[:,Q-3]-2*radon[:,Q-2] +radon[:,Q-1]) / (h*h)
    return -dr_radon

def adjoint_imaging_operator(radon, p, q, p_rec) -> NDArray:
    """
    Berechnet den adjungierten Operator M* der zirkulären Mittel-Radon-Transformation.
    
    Parameter:
        radon : NDArray
            2D-Datenarray der Radon-Messwerte, Form (len(angles), len(radii)).
        angles : NDArray
            Array der Winkel phi, über die die Kreise definiert sind.
        radii : NDArray
            Array der Radien der Kreise.
    
    Rückgabe:
        image : NDArray
            Das rekonstruierte Bild (als Rückprojektion von M*).
    """
    angles = np.linspace(0,2*np.pi, p)
    # interpolator = RegularGridInterpolator((angles, radii), radon, bounds_error=False, fill_value=None)
    result = np.zeros((p_rec,p_rec))

    #Numerische Integration mit Trapezregel
    dphi = (2 * np.pi) / p

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing...", total=p_rec)
        for x1 in range(p_rec):
            for x2 in range(p_rec):
                y1 = 2*x1 / p_rec - 1
                y2 = 2*x2 / p_rec - 1
                if np.sqrt(y1**2 + y2**2) <= 1:
                    for phi_idx, phi in enumerate(angles):
                        r = np.sqrt((y1 + np.sin(phi))**2 + (y2 - np.cos(phi))**2)
                        if r != 0 and r <= 2:
                            r_idx = r / 2 *(q - 1)
                            g_int = map_coordinates(radon, [[phi_idx], [r_idx]], order=1, mode='nearest')[0]
                            result[x1, x2] += 1 / (2*np.pi*r) * g_int * dphi
            progress.update(task, advance=1)
            # print("Progress reconstruction: ", x1/p_rec*100, "%")
    return result / (np.max(np.abs(result))) 

def compute_pixel(args):
    """ Berechnet den adjungierten Operator für ein einzelnes Pixel (x1, x2). """
    radon, angles, p, q, p_rec, dphi, x1, x2 = args
    y1 = 2*x1 / p_rec - 1
    y2 = 2*x2 / p_rec - 1
    result_value = 0

    if np.sqrt(y1**2 + y2**2) <= 1:
        for i, phi in enumerate(angles):
            r = np.sqrt((y1 - np.cos(phi))**2 + (y2 - np.sin(phi))**2)
            if r != 0 and r <= 2:
                phi_idx = i / (p - 1) * (p - 1)
                r_idx = r / 2 * (q - 1)
                g_int = map_coordinates(radon, [[phi_idx], [r_idx]], order=1, mode='nearest')[0]
                result_value += 1 / (2*np.pi*r) * g_int * dphi

    return (x1, x2, result_value)

def adjoint_imaging_operator_parallel(radon, p, q, p_rec):
    angles = np.linspace(0, 2*np.pi, p, endpoint=False)
    dphi = (2 * np.pi) / p
    result = np.zeros((p_rec, p_rec))

    # Liste aller Argumente für `compute_pixel`
    args_list = [(radon, angles, p, q, p_rec, dphi, x1, x2) 
                 for x1 in range(p_rec) for x2 in range(p_rec)]

    # Parallel rechnen mit multiprocessing
    with Pool() as pool:
        for x1, x2, value in tqdm(pool.imap_unordered(compute_pixel, args_list), total=len(args_list), desc="Processing"):
            result[x1, x2] = value

    return result / np.max(result)

def adjoint_imaging_operator_parallel_2(radon, p, q, p_rec) -> np.ndarray:
    angles = np.linspace(0, 2*np.pi, p)
    radii = np.linspace(0, 2, q)
    result = np.zeros((p_rec, p_rec))
    dphi = (2 * np.pi) / p

    def process_pixel(x1):
        row_result = np.zeros(p_rec)
        for x2 in range(p_rec):
            y1 = 2 * x1 / p_rec - 1
            y2 = 2 * x2 / p_rec - 1
            if np.sqrt(y1**2 + y2**2) <= 1:
                for i, phi in enumerate(angles):
                    r = np.sqrt((y1 + np.cos(phi))**2 + (y2 - np.sin(phi))**2)
                    if r != 0 and r <= 2:
                        phi_idx = i / (p - 1) * (p - 1)
                        r_idx = r / 2 * (q - 1)
                        g_int = map_coordinates(radon, [[phi_idx], [r_idx]], order=1, mode='nearest')[0]
                        row_result[x2] += 1 / (2*np.pi*r) * g_int * dphi
        return row_result

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing...", total=p_rec)
        results = Parallel(n_jobs=-1)(delayed(process_pixel)(x1) for x1 in range(p_rec))
        for x1, row in enumerate(results):
            result[x1, :] = row
            progress.update(task, advance=1)

    return result / np.max(result)

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
    ellipse = np.zeros((dim, dim), dtype=int)

    for i in range(0, 2 * p + 1):
        x = -1 + i * h - cx  # x-coordinate in the grid
        for j in range(0, 2 * p + 1):
            y = -1 + j * h + cy  # y-coordinate in the grid
            if (x*x/(a*a) + y*y/(b*b)) <= 1:
                ellipse[j,i] = 1        
    return ellipse
