import numpy as np
import matplotlib.pyplot as plt
from problems.problem_ct import *
from matplotlib.patches import Circle

pfad_reco = r'Data\PAT\20250510_123939_reconstruction.npy'
pfad_orig = r'Data\PAT\20250510_123939original.npy'
orig = np.load(pfad_orig)
reconstruction = np.load(pfad_reco)

dash_pattern = (0, (10, 10))
kreis = Circle((0, 0), 1, edgecolor='black', facecolor='none', linewidth=1, linestyle=dash_pattern)
kreis2 = Circle((0, 0), 1, edgecolor='black', facecolor='none', linewidth=1, linestyle=dash_pattern)
kreis_roi = Circle((0, 0), 0.5, edgecolor='green', facecolor='none', linewidth=1, linestyle=dash_pattern)
kreis_roi2 = Circle((0, 0), 0.5, edgecolor='green', facecolor='none', linewidth=1, linestyle=dash_pattern)
fig, axs = plt.subplots(1,2, figsize=(12, 6))
# Original Picture
im1 = axs[0].imshow(orig, extent=[-1, 1, -1, 1], aspect='equal', cmap='gray_r', origin='upper', interpolation='nearest')
axs[0].set_title('Originalbild')
axs[0].add_patch(kreis)
axs[0].add_patch(kreis_roi)
fig.colorbar(im1, ax=axs[0], shrink=0.7)

# Reconstruction
im2 = axs[1].imshow(reconstruction, extent=[-1, 1, -1, 1], aspect='equal', cmap='gray_r', origin='upper')
axs[1].set_title('Rekonstruktion')
axs[1].add_patch(kreis2)
axs[1].add_patch(kreis_roi2)
fig.colorbar(im2, ax=axs[1], shrink=0.7)
plt.tight_layout()
plt.savefig("_figure.svg", bbox_inches='tight')
plt.show()