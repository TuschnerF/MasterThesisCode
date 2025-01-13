# This is an implementation of the reconstruction operator of the 
# circular mean Radon transform used in PAT or TAT

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from  problems.problem import Problem

class Problem_ROI(Problem):
    # Constructor
    def __init__(self):
        pass
    
    def imaging_operator(self, image: NDArray, angles : NDArray, radii) -> NDArray:
        pass