import numpy as np
import matplotlib.pyplot as plt
from problems.problem_pat import *
import time

q=500
p=1000
p_rec=500
orig = draw_ellipse(0.2, 0.2, 0.0, 0.0, 400) 
# orig = draw_rectangle(0.2,0.2,1/4*np.pi,0,0.1,200)
start_sinogram = time.perf_counter()
sinogram = imaging_operator(orig, p, q)
sinogram_filtered = sinogram.copy()
#filter
for i in range(p):
    if i<=p/2:
        sinogram_filtered[i,:]=0
start_reconstruction = time.perf_counter()
print("Sinogramm done", start_reconstruction-start_sinogram)
# reconstruction = adjoint_imaging_operator(sinogram,q,p, p_rec)
# print("Reconstruction done")
reconstruction_filtered = adjoint_imaging_operator(sinogram_filtered,p,q, p_rec)
end_reconstruction = time.perf_counter()
print("Sinogramm done", end_reconstruction-start_reconstruction)

fig, axs = plt.subplots(1, 2, figsize=(16, 10))
# Zeige das erste Bild im ersten Subplot
im1 = axs[0].imshow(orig, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray', origin='upper')
axs[0].set_title('Original Image')
axs[0].set_aspect('equal')
fig.colorbar(im1, ax=axs[0])

# Zeige das zweite Bild im zweiten Subplot
im2 = axs[1].imshow(reconstruction_filtered, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray', origin='upper')
axs[1].set_title('Reconstruction')
axs[1].set_aspect('equal')
fig.colorbar(im2, ax=axs[1])

# im3 = axs[2].imshow(reconstruction_filtered, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray', origin='upper')
# axs[2].set_title('Reconstruction from filtered Data 0-pi')
# axs[2].set_aspect('equal')
# fig.colorbar(im3, ax=axs[2])

# Zeige die Figur
plt.tight_layout()
plt.show()
plt.savefig("PAT.png")
# image = np.zeros([4,4])
# image[0][1] = 1
# test = map_coordinates(image, [[0, 0.5], [0.5, 0.5]], order=1)
# print(test)
# plt.imshow(image, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray', origin='upper')
# plt.show()
