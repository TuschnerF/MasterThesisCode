import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import cv2

# Add a circle to a squared Matrix with coordinate represantation of D=unitdisc
def add_circle_D(image: NDArray, pos: NDArray = np.array([0, 0]), radius: float = 1, value: float = 1) -> NDArray:
    n, m = image.shape
    if n != m:
        raise ValueError("image not squared")
    
    center = np.array([n // 2, m // 2])
    _pos = center + pos*(n // 2)
    _radius = int(radius * (n // 2))
    image = cv2.circle(image, _pos, _radius, value, -1)
    return image

# Add a rectangle to a squared Matrix with coordinate represantation of D=unitdisc
def add_rectangle_D(image: NDArray, start_pos: NDArray = np.array([0, 0]), end_pos: NDArray = np.array([0, 0]), value: float = 1) -> NDArray:
    n, m = image.shape
    if n != m:
        raise ValueError("image not squared")
    
    center = np.array([n // 2, m // 2])
    _startpos = center + start_pos*(n // 2)
    _endpos = center + end_pos*(n // 2)
    image = cv2.rectangle(image, _startpos.astype(int), _endpos.astype(int), value, -1)
    return image