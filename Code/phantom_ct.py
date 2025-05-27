from skimage.data import shepp_logan_phantom
import matplotlib.pyplot as plt
from skimage.transform import resize
from problems.problem_lap import *
import numpy as np
import os
from datetime import datetime

def write_output_file(id, safe_dir, variables, calc_filter_time, calc_rec_time):
    with open(os.path.join(safe_dir, str(id)+"_logfile"+".txt"), "w") as file: 
        for i in range(len(variables)):
            if len(variables[i]) == 2:
                file.write(str(variables[i][0]) + " = " + str(variables[i][1]) +"\n")
        file.write("Elapsed Time:"+"\n")
        file.write("Calculation Filter:" + str(calc_filter_time)+"\n")
        file.write("Calculation Reconstruction:" + str(calc_rec_time)+"\n")

p=500
q=500
p_rec=1000
dim = 2*q+1
save = True
safe_dir = "Data/LAP_Phantom"
os.makedirs(safe_dir, exist_ok=True)
n=500
lr = 0.5
la = 0.0
cutoff = False

start_sinogram = time.perf_counter()
orig = shepp_logan_phantom()
# orig = resize(orig, (1000, 1000))
# orig = draw_ellipse(0.25,0.25,0.0,0.0, 1000) 
# orig = draw_rectangle(0.3, 0.3, np.pi/8, 0.5, 0.0, 1000) 
rows, columns = orig.shape

start_reconstruction = time.perf_counter()
sinogram = imaging_operator(orig, p, q)
# sinogram = projellipse(0.25,0.25,0.0,0.0, q, p)

print("Sinogramm done", start_reconstruction-start_sinogram)
start_filter = time.perf_counter()
eta = filter(sinogram, q, p, n, lr, la, cutoff)
end_filter = time.perf_counter()

print("S2: Reconstruction for ", p_rec*p_rec, " points")
start_reconstruction = time.perf_counter()
reconstruction = filtered_backprojection_paralell_paralell(eta, q, p, p_rec)
print("Reconstruction finished")
end_reconstruction = time.perf_counter()
calc_filter_time = end_filter-start_filter
calc_rec_time = end_reconstruction-start_reconstruction

# -------------------------------- Plotten --------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
im1 = axs[0].imshow(orig, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray_r', origin='upper', vmin=-0.1, vmax=1.1)
axs[0].set_title('Originalbild')
axs[0].set_aspect('equal')
fig.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(reconstruction, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray_r', origin='upper', vmin=-0.1, vmax=1.1)
axs[1].set_title('Rekonstruktion')
axs[1].set_aspect('equal')
fig.colorbar(im2, ax=axs[1])

# im3 = axs[2].imshow(reconstruction_filtered, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray_r', origin='upper')
# axs[2].set_title('smoothed Reconstruction from filtered Data 0-pi ')
# axs[2].set_aspect('equal')
# fig.colorbar(im3, ax=axs[2])

# Zeige die Figur
plt.tight_layout()
if save == True:
    id = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_output_file(id, safe_dir,  [["Typ", "Phantom"], ["q",q], ["p",p], ["p_rec",p_rec]], calc_filter_time, calc_rec_time)
    plt.savefig(os.path.join(safe_dir, str(id)+"_figure.svg"), bbox_inches='tight')
    np.save(os.path.join(safe_dir, str(id)+"original"), orig)
    np.save(os.path.join(safe_dir, str(id)+"_reconstruction"), reconstruction)
                                      
plt.show()
