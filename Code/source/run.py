from problems.problem_lap import *
import matplotlib.pyplot as plt
import numpy as np
import os

def write_output_file(id, safe_dir, variables, calc_filter_time, calc_rec_time):
    with open(os.path.join(safe_dir, str(id)+"_logfile"+".txt"), "w") as file: 
        for i in range(len(variables)):
            if len(variables[i]) == 2:
                file.write(str(variables[i][0]) + " = " + str(variables[i][1]) +"\n")
        file.write("Elapsed Time:"+"\n")
        file.write("Calculation Filter:" + str(calc_filter_time)+"\n")
        file.write("Calculation Reconstruction:" + str(calc_rec_time)+"\n")

def run_loop_rectangle( lr = 1.0, la = 0.0,  cutoff = False):
    id = 0
    ur_list = [0.2]
    vr_list = [0.2]
    phi_list = [0.0]
    cx_list = [0.25]
    cy_list = [0.25]
    q_list = [40]
    p_list = [80]
    n_list = [1500]
    p_rec_list = [101]

    safe_dir = "Data"  # Define the directory
    os.makedirs(safe_dir, exist_ok=True)

    for ur in ur_list:
        for vr in vr_list:
            for phi in phi_list:
                for cx in cx_list:
                    for cy in cy_list:
                        for q in q_list:
                            for p in p_list:
                                orig = draw_rectangle(ur, vr, phi, cx, cy, p) 
                                sinogram = projrec(ur, vr, phi, cx, cy, q, p)
                                for n in n_list:
                                    start_filter = time.perf_counter()
                                    eta = filter(sinogram, q, p, n, lr, la, cutoff)
                                    end_filter = time.perf_counter()
                                    
                                    for p_rec in p_rec_list:
                                        print("S2: Reconstruction for ", p_rec*p_rec, " points")
                                        start_reconstruction = time.perf_counter()
                                        reconstruction = filtered_backprojection_paralell(eta, q, p, p_rec)
                                        print("Reconstruction finished")
                                        end_reconstruction = time.perf_counter()
                                        calc_filter_time = end_filter-start_filter
                                        calc_rec_time = end_reconstruction-start_reconstruction

                                        write_output_file(id, safe_dir,  [["ur",ur], ["vr",vr], ["phi",phi], ["cx",cx], ["cy",cy], ["q",q], ["p",p], ["n",n], ["p_rec",p_rec]], calc_filter_time, calc_rec_time)

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
                                        plt.savefig(os.path.join(safe_dir, str(id)+"_figure.png"))
                                        #plt.show()
                                        
                                        print("Elapsed Time:")
                                        print("Calculation Filter:", calc_filter_time)
                                        print("Calculation Reconstruction:", calc_rec_time)
                                        id+=1

    