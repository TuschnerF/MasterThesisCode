import numpy as np
import matplotlib.pyplot as plt
from problems.problem_pat import *
import time
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


p_rec=100
p = 50
q = 50
cutoff = False
orig = draw_ellipse(0.2, 0.2, 0.0, 0.0, 1000) 
# orig = draw_rectangle(0.2,0.2,5/180*np.pi,0,0.3,200)

# filtered data: use only data in [a,b]
a = 0
b = np.pi
start_sinogram = time.perf_counter()
sinogram = imaging_operator(orig, p, q)
sinogram_filtered = sinogram.copy()
save = True
safe_dir = "Data/PAT"
os.makedirs(safe_dir, exist_ok=True)
#filter
for i in range(p):
    # if i>=p/2:
    #     sinogram_filtered[i,:]=0
    phi = i * 2 * np.pi / p
    if phi >= b:
        sinogram[i,:] = 0

start_reconstruction = time.perf_counter()
print("Sinogramm done", start_reconstruction-start_sinogram)
calc_filter_time = start_reconstruction-start_sinogram
reconstruction = adjoint_imaging_operator_parallel(sinogram,q,p, p_rec)
end_reconstruction = time.perf_counter()
print("Reconstruction done", end_reconstruction-start_reconstruction)
calc_rec_time = end_reconstruction-start_reconstruction


# -------------------------------- Plotten --------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
im1 = axs[0].imshow(orig, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray_r', origin='upper')
axs[0].set_title('Originalbild')
axs[0].set_aspect('equal')
# fig.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(reconstruction, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray_r', origin='upper')
axs[1].set_title('Rekonstruktion')
axs[1].set_aspect('equal')
# fig.colorbar(im2, ax=axs[1])

# Zeige die Figur
plt.tight_layout()
if save == True:
    id = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_output_file(id, safe_dir,  [["Typ", "Ellipse(0.2,0.2)"], ["a",a], ["b",b], ["q",q], ["p",p], ["p_rec",p_rec]], calc_filter_time, calc_rec_time)
    plt.savefig(os.path.join(safe_dir, str(id)+"_figure.svg"), bbox_inches='tight')
    np.save(os.path.join(safe_dir, str(id)+"original"), orig)
    np.save(os.path.join(safe_dir, str(id)+"_reconstruction"), reconstruction)
                                      
plt.show()
