# import matplotlib.pyplot as plt
from problems.problem_lap import *
import numpy as np
import matplotlib.pyplot as plt

orig = np.zeros((10,10))
orig[0,0]=1
orig[0,-1]=-1
plt.figure(figsize=(10, 6))
plt.imshow(np.transpose(orig), extent=[-1, 1, -1,1], aspect='auto', cmap='gray', origin='lower')
plt.colorbar()
plt.show()