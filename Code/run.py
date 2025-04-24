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

def run_loop_rectangle( lr = 1.0, la = 0.0,  cutoff = True):
    id = 0
    ur_list = [0.2]
    vr_list = [0.2]
    phi_list = [np.pi/4]
    cx_list = [0.0]
    cy_list = [0.0]
    q_list = [300] #300
    p_list = [300] #300
    n_list = [1500]
    p_rec_list = [300]
    save = True

    safe_dir = ""
    if lr != 1.0:
        safe_dir = "Data/ROI"
    if la != 0.0:
        safe_dir = "Data/LA"
    os.makedirs(safe_dir, exist_ok=True)

    for ur in ur_list:
        for vr in vr_list:
            for phi in phi_list:
                for cx in cx_list:
                    for cy in cy_list:
                        for q in q_list:
                            for p in p_list:
                                orig = draw_rectangle(ur, vr, phi, cx, cy, 1000) 
                                # orig = draw_ellipse(0.25,0.25,0.0,0.0, 1000) 
                                draw_roi(orig,lr)
                                sinogram = projrec(ur, vr, phi, cx, cy, q, p)
                                # sinogram = projellipse(0.25, 0.25, 0.0, 0.0, q, p)
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
                                        im1 = axs[0].imshow(orig, extent=[-1, 1, -1, 1], aspect='equal', cmap='gray', origin='upper', interpolation='nearest', vmin=-0.1, vmax=1.1)
                                        axs[0].set_title('Originalbild')
                                        fig.colorbar(im1, ax=axs[0], shrink=0.7)

                                        # Reconstruction
                                        im2 = axs[1].imshow(reconstruction, extent=[-1, 1, -1, 1], aspect='equal', cmap='gray', origin='upper', vmin=-0.1, vmax=1.1)
                                        axs[1].set_title('Rekonstruktion')
                                        fig.colorbar(im2, ax=axs[1], shrink=0.7)

                                        plt.tight_layout()
                                        # Save
                                        if save == True:
                                            id = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            write_output_file(id, safe_dir,  [["ur",ur], ["vr",vr], ["phi",phi], ["cx",cx], ["cy",cy], ["q",q], ["p",p], ["n",n], ["p_rec",p_rec]], calc_filter_time, calc_rec_time)
                                            plt.savefig(os.path.join(safe_dir, str(id)+"_figure.svg"))
                                            np.save(os.path.join(safe_dir, str(id)+"original"), orig)
                                            np.save(os.path.join(safe_dir, str(id)+"_reconstruction"), reconstruction)
                                        plt.show()
                                        
                                        print("Elapsed Time:")
                                        print("Calculation Filter:", calc_filter_time)
                                        print("Calculation Reconstruction:", calc_rec_time)

def run_loop_ellipse( lr = 1.0, la = 0.0,  cutoff = True):
    id = 0
    a = 0.2
    b = 0.2
    cx = 0.3
    cy = 0.0
    q = 300 #300
    p = 300 #300
    n = 1500
    p_rec = 400
    save = True

    safe_dir = ""
    if lr != 1.0:
        safe_dir = "Data/ROI"
    if la != 0.0:
        safe_dir = "Data/LA"
    os.makedirs(safe_dir, exist_ok=True)

    orig = draw_ellipse(a, b, cx, cy, 1000) 
    if lr != 1.0:
        draw_roi(orig,lr)
    sinogram = projellipse(a, b, cx, cy, q, p)

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
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Original Picture
    im1 = axs[0].imshow(orig, extent=[-1, 1, -1, 1], aspect='equal', cmap='gray', origin='upper', interpolation='nearest', vmin=-0.1, vmax=1.1)
    axs[0].set_title('Originalbild')
    fig.colorbar(im1, ax=axs[0], shrink=0.7)

    # Reconstruction
    im2 = axs[1].imshow(reconstruction, extent=[-1, 1, -1, 1], aspect='equal', cmap='gray', origin='upper', vmin=-0.1, vmax=1.1)
    axs[1].set_title('Rekonstruktion')
    fig.colorbar(im2, ax=axs[1], shrink=0.7)

    plt.tight_layout()
    # Save
    if save == True:
        id = datetime.now().strftime("%Y%m%d_%H%M%S")
        write_output_file(id, safe_dir,  [["a",a], ["b",b], ["cx",cx], ["cy",cy], ["q",q], ["p",p], ["n",n], ["p_rec",p_rec], ["lr",lr], ["la",la]], calc_filter_time, calc_rec_time)
        plt.savefig(os.path.join(safe_dir, str(id)+"_figure.svg"))
        np.save(os.path.join(safe_dir, str(id)+"original"), orig)
        np.save(os.path.join(safe_dir, str(id)+"_reconstruction"), reconstruction)
    plt.show()
    
    print("Elapsed Time:")
    print("Calculation Filter:", calc_filter_time)
    print("Calculation Reconstruction:", calc_rec_time)

# run_loop_rectangle(lr=0.5)
run_loop_ellipse(lr=0.5, cutoff=False)
