# This is an implementation of the reconstruction operator of the 
# circular mean Radon transform used in PAT or TAT

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from  Problems.problem import Problem

class Problem_PAT(Problem):
    # Constructor
    def __init__(self):
        pass
    
    def imaging_operator(self, image: NDArray, angles : NDArray, radii) -> NDArray:
        # Implementation of the circular mean Radon transform
        # image: 2D image
        # angles: Array with all phi for points on the sphere: Theta(phi)
        # radii: Array of all r for all circles
        radon = np.zeros((len(angles), len(radii)))
        rows, columns = image.shape
        center = (rows // 2, columns // 2)  # Assume image is centered
        n_points_circle = 360
        for i,phi in enumerate(angles):
            for j, r in enumerate(radii):
                psi = np.linspace(0, 2*np.pi, n_points_circle, endpoint=False)
                x_circle = (2 + r*np.cos(phi) + r*np.cos(psi)) * columns / 2
                y_circle = (2 + r*np.sin(phi) + r*np.sin(psi)) * rows / 2
                
                # bilinear interpolation of the y,y values in the imae
                sampled_values = map_coordinates(image, [y_circle, x_circle], order=1, mode='constant')
                
                # Compute the average value over the sampled points (arc length measure)    	
                radon[i, j] = np.mean(sampled_values)
        return radon



