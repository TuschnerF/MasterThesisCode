# import matplotlib.pyplot as plt
from problems.problem_lap import *
import numpy as np
import matplotlib.pyplot as plt
print(int(5.8))
ur = 0.2
vr = 0.1
phi = 0
cx = 0.1
cy = 0.0
q = 100
p = 320
n = 10000 # regularization parameter

orig = rectangle(ur, vr, phi, cx, cy, p) 
# plt.figure(figsize=(10, 6))
# plt.imshow(orig, extent=[-1, 1, -1,1], aspect='auto', cmap='gray', origin='lower')
# plt.colorbar()
# plt.show()
# print(orig)

sinogram = projrec(ur, vr, phi, cx, cy, q, p)
reconstruction = filtered_backprojection_paralell(sinogram, q, p, 101, n)

# plt.figure(figsize=(10, 6))
# plt.imshow(reconstruction, extent=[-1, 1, -1,1], aspect='auto', cmap='gray', origin='lower')
# plt.colorbar()
# plt.show()

fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Zeige das erste Bild im ersten Subplot
im1 = axs[0].imshow(orig, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray', origin='lower')
axs[0].set_title('Original Image')
fig.colorbar(im1, ax=axs[0])

# Zeige das zweite Bild im zweiten Subplot
im2 = axs[1].imshow(reconstruction, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray', origin='lower')
axs[1].set_title('Reconstruction')
fig.colorbar(im2, ax=axs[1])

# Zeige die Figur
plt.tight_layout()
plt.show()