import numpy as np
import matplotlib.pyplot as plt
import cv2
from source.data import *

a = np.zeros((500,500))
# a = cv2.rectangle(a, [100,100], [200,200],1, -1)
a = add_rectangle_D(a, np.array([0.0]), np.array([0.5,0.5]), 0.5)
a = add_circle_D(a, np.array([0,0]), 0.25, 1)
plt.imshow(a, cmap='gray')
plt.colorbar()
plt.show()