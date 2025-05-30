# import matplotlib.pyplot as plt
from problems.problem_lap import *
import numpy as np
import matplotlib.pyplot as plt
import os
from run import *
# save_dir = "Data"  # Define the directory
# os.makedirs(save_dir, exist_ok=True) 

# ur = 0.2
# vr = 0.2
# phi = 0.0
# cx = 0.25
# cy = 0.25
# q = 40
# p = 80
# n = 1500 # regularization parameter
# p_rec = 101

# orig = draw_rectangle(ur, vr, phi, cx, cy, p) 
# sinogram = projrec(ur, vr, phi, cx, cy, q, p)
# reconstruction = filtered_backprojection_paralell(sinogram, q, p, p_rec, n, lr=0.5, cutoff = True)

# fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# # Zeige das erste Bild im ersten Subplot
# im1 = axs[0].imshow(orig, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray', origin='upper')
# axs[0].set_title('Original Image')
# fig.colorbar(im1, ax=axs[0])

# # Zeige das zweite Bild im zweiten Subplot
# im2 = axs[1].imshow(reconstruction, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray', origin='upper')
# axs[1].set_title('Reconstruction')
# fig.colorbar(im2, ax=axs[1])

# # Zeige die Figur
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "figure.png"))
# plt.show()

run_loop_rectangle(la = np.pi/4)