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


# q=500
# p=500
p_rec=500
p = 2000
q = 250
cutoff = True
orig = draw_ellipse(0.2, 0.2, 0.0, 0.0, 1000) 
a = 0
b = np.pi
# orig = draw_rectangle(0.2,0.2,5/180*np.pi,0,0.3,200)
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
    # if cutoff == True:
    #     eps = (b-a)/6
    #     if phi>a and phi<a+eps:
    #         sinogram_filtered[:,i] *= cut_off_function(a+eps-phi, eps)
    #     if phi > b-eps and phi < b:
    #         sinogram_filtered[:,i] *= cut_off_function(phi-b+eps, eps)


start_reconstruction = time.perf_counter()
print("Sinogramm done", start_reconstruction-start_sinogram)
calc_filter_time = start_reconstruction-start_sinogram
reconstruction = adjoint_imaging_operator_parallel_2(sinogram,q,p, p_rec)
# print("Reconstruction done")
# reconstruction_filtered = adjoint_imaging_operator(sinogram_filtered,p,q, p_rec)
# reconstruction_filtered = adjoint_imaging_operator_parallel_2(sinogram_filtered,p,q, p_rec)
end_reconstruction = time.perf_counter()
print("Reconstruction done", end_reconstruction-start_reconstruction)
calc_rec_time = end_reconstruction-start_reconstruction


# -------------------------------- Plotten --------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 8))
im1 = axs[0].imshow(orig, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray_r', origin='upper')
axs[0].set_title('Original Image')
axs[0].set_aspect('equal')
# fig.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(reconstruction, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray_r', origin='upper')
axs[1].set_title('Reconstruction from filtered Data 0-pi')
axs[1].set_aspect('equal')
# fig.colorbar(im2, ax=axs[1])

# im3 = axs[2].imshow(reconstruction_filtered, extent=[-1, 1, -1, 1], aspect='auto', cmap='gray_r', origin='upper')
# axs[2].set_title('smoothed Reconstruction from filtered Data 0-pi ')
# axs[2].set_aspect('equal')
# fig.colorbar(im3, ax=axs[2])

# Zeige die Figur
plt.tight_layout()
if save == True:
    id = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_output_file(id, safe_dir,  [["Typ", "Ellipse(0.2,0.2)"], ["a",a], ["b",b], ["q",q], ["p",p], ["p_rec",p_rec]], calc_filter_time, calc_rec_time)
    plt.savefig(os.path.join(safe_dir, str(id)+"_figure.svg"))
    np.save(os.path.join(safe_dir, str(id)+"original"), orig)
    np.save(os.path.join(safe_dir, str(id)+"_reconstruction"), reconstruction)
                                      

plt.show()
