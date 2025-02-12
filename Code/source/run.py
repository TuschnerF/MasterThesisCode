from problems.problem_lap import *
import matplotlib.pyplot as plt
import numpy as np
import os

def write_output_file(filename: str, safe_dir: str, id, variables):
    os.makedirs(safe_dir, exist_ok=True) 
    with open(os.path.join(safe_dir, str(id)+"_"+filename+".txt"), "w") as file: 
        for i in range(len(variables)):
            if len(variables[i]) == 2:
                file.write(str(variables[i][0]) + " = " + str(variables[i][1]) +"\n")

def run_loop_rectangle():
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

    for 