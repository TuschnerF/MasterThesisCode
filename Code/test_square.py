# import matplotlib.pyplot as plt
from problems.problem_lap import *
import numpy as np
import matplotlib.pyplot as plt

ur = 2/3 - np.sqrt(2)/3
vr = 2/3 - np.sqrt(2)/3
phi = 0.0
cx = np.sqrt(2)/3
cy = np.sqrt(2)/3
q = 250
p = 500
n = 1500 # regularization parameter
p_rec = 151

orig = draw_rectangle(ur, vr, phi, cx, cy, p) 
sinogram = projrec(ur, vr, phi, cx, cy, q, p)
reconstruction = filtered_backprojection_paralell(sinogram, q, p, p_rec, n, la=np.pi/16)

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Zeige das erste Bild im ersten Subplot
im1 = axs[0].imshow(orig, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray', origin='upper')
axs[0].set_title('Original Image')
fig.colorbar(im1, ax=axs[0])

# Zeige das zweite Bild im zweiten Subplot
im2 = axs[1].imshow(reconstruction, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray', origin='upper')
axs[1].set_title('Reconstruction')
fig.colorbar(im2, ax=axs[1])

# Zeige die Figur
plt.tight_layout()
plt.show()