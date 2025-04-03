from problems.problem_lap import *
import matplotlib.pyplot as plt
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

def run_loop_rectangle( lr = 0.5, la = 0.0,  cutoff = False):
    id = 0
    ur_list = [0.25]
    vr_list = [0.25]
    phi_list = [0.0]
    cx_list = [0.25]
    cy_list = [0.5]
    q_list = [100]
    p_list = [100]
    n_list = [1500]
    p_rec_list = [100]
    save = False

    safe_dir = "Data/ROI"  # Define the directory
    # safe_dir = "Data/LA"  # Define the directory
    os.makedirs(safe_dir, exist_ok=True)

    for ur in ur_list:
        for vr in vr_list:
            for phi in phi_list:
                for cx in cx_list:
                    for cy in cy_list:
                        for q in q_list:
                            for p in p_list:
                                orig = draw_rectangle(ur, vr, phi, cx, cy, 1000) 
                                draw_roi(orig,lr)
                                sinogram = projrec(ur, vr, phi, cx, cy, q, p)
                                # sinogram = projellipse(0.2, 0.2, cx, cy, q, p)
                                for n in n_list:
                                    start_filter = time.perf_counter()
                                    eta = filter(sinogram, q, p, n, lr, la, cutoff)
                                    end_filter = time.perf_counter()

                                    for p_rec in p_rec_list:
                                        print("S2: Reconstruction for ", p_rec*p_rec, " points")
                                        start_reconstruction = time.perf_counter()
                                        reconstruction = filtered_backprojection_paralell_paralell(eta, q, p, p_rec)
                                        print("Reconstruction finished")
                                        end_reconstruction = time.perf_counter()
                                        calc_filter_time = end_filter-start_filter
                                        calc_rec_time = end_reconstruction-start_reconstruction
                                        
                                        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

                                        # Original Picture
                                        im1 = axs[0].imshow(orig, extent=[-1, 1, -1, 1], aspect='equal', cmap='turbo', origin='upper', interpolation='nearest', vmin=-0.1, vmax=1.1)
                                        axs[0].set_title('Original Image')
                                        fig.colorbar(im1, ax=axs[0], shrink=0.7)

                                        # Reconstruction
                                        im2 = axs[1].imshow(reconstruction, extent=[-1, 1, -1, 1], aspect='equal', cmap='turbo', origin='upper', vmin=-0.1, vmax=1.1)
                                        axs[1].set_title('Reconstruction')
                                        fig.colorbar(im2, ax=axs[1], shrink=0.7)

                                        plt.tight_layout()
                                        # Save
                                        if save == True:
                                            id = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            write_output_file(id, safe_dir,  [["ur",ur], ["vr",vr], ["phi",phi], ["cx",cx], ["cy",cy], ["q",q], ["p",p], ["n",n], ["p_rec",p_rec]], calc_filter_time, calc_rec_time)
                                            plt.savefig(os.path.join(safe_dir, str(id)+"_figure.svg"))
                                            np.save(os.path.join(safe_dir, str(id)+"original.svg"), orig)
                                            np.save(os.path.join(safe_dir, str(id)+"_reconstruction.svg"), reconstruction)
                                        plt.show()
                                        
                                        print("Elapsed Time:")
                                        print("Calculation Filter:", calc_filter_time)
                                        print("Calculation Reconstruction:", calc_rec_time)

                                        # fig, axs = plt.subplots(1, 2, figsize=(12, 6))

                                        # # Zeige das erste Bild im ersten Subplot
                                        # im1 = axs[0].imshow(orig, extent=[-1, 1, -1, 1], aspect='equal', cmap='gray', origin='upper', interpolation='nearest')
                                        # axs[0].set_title('Original Image')
                                        # fig.colorbar(im1, ax=axs[0], shrink=0.7)

                                        # # Zeige das zweite Bild im zweiten Subplot
                                        # im2 = axs[1].imshow(reconstruction, extent=[-1, 1, -1, 1], aspect='equal', cmap='gray', origin='upper')
                                        # axs[1].set_title('Reconstruction')
                                        # fig.colorbar(im2, ax=axs[1], shrink=0.7)

                                        # # Zeige die Figur
                                        # plt.tight_layout()
                                        # plt.savefig(os.path.join(safe_dir, str(id)+"_figure_gray.svg"))
                                        # # plt.show()
                                       

run_loop_rectangle(lr=0.5)