import numpy as np
import matplotlib.pyplot as plt
from problems.problem_pat import *
import time

q=100
p=100
p_rec=100
orig = draw_ellipse(0.2, 0.2, 0.0, 0.0, 1000) 
# orig = draw_rectangle(0.2,0.2,5/180*np.pi,0,0.3,200)
start_sinogram = time.perf_counter()
sinogram = imaging_operator(orig, p, q)
sinogram_filtered = sinogram.copy()
#filter
for i in range(p):
    if i>=p/2:
        sinogram_filtered[i,:]=0
start_reconstruction = time.perf_counter()
print("Sinogramm done", start_reconstruction-start_sinogram)
# reconstruction = adjoint_imaging_operator(sinogram,q,p, p_rec)
# print("Reconstruction done")
# reconstruction_filtered = adjoint_imaging_operator(sinogram_filtered,p,q, p_rec)
reconstruction_filtered = adjoint_imaging_operator_parallel_2(sinogram_filtered,p,q, p_rec)
end_reconstruction = time.perf_counter()
print("Sinogramm done", end_reconstruction-start_reconstruction)



# -------------------------------- Plotten --------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 8))
im1 = axs[0].imshow(orig, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray_r', origin='upper')
axs[0].set_title('Original Image')
axs[0].set_aspect('equal')
fig.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(reconstruction_filtered, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray_r', origin='upper')
axs[1].set_title('Reconstruction')
axs[1].set_aspect('equal')
fig.colorbar(im2, ax=axs[1])

# Zeige die Figur
plt.tight_layout()
plt.savefig("PAT.png")
plt.show()
