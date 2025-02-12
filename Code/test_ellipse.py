# import matplotlib.pyplot as plt
from problems.problem_lap import *
import numpy as np
import matplotlib.pyplot as plt

a = 0.3
b = 0.3
# phi = 0.0
cx = 0
cy = 0
q = 100
p = 200
n = 1500 # regularization parameter
p_rec = 101

orig = draw_ellipse(a, b, cx, cy, p) 
sinogram = projellipse(a, b, cx, cy, q, p)
reconstruction = filtered_backprojection_paralell(sinogram, q, p, p_rec, n , la=np.pi/4)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Zeige das erste Bild im ersten Subplot
im1 = axs[0].imshow(orig, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray', origin='upper')
axs[0].set_title('Original Image')
axs[0].set_aspect('equal')
fig.colorbar(im1, ax=axs[0])

# Zeige das zweite Bild im zweiten Subplot
im2 = axs[1].imshow(reconstruction, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray', origin='upper')
axs[1].set_title('Reconstruction')
axs[1].set_aspect('equal')
fig.colorbar(im2, ax=axs[1])

# Zeige die Figur
plt.tight_layout()
plt.show()