import os
import numpy as np

Gamma_list = np.linspace(0.0, 5.0, 6)
Omega = 1.0
E = 1.0
mu = 0.0

tmax = 80.0
nt = 1001
T_list = np.linspace(0.0, 10.0, 101)
n_omega = 2049

measure = ['ALL']

folder_name = 'gamma_scan'
folder_directory = os.path.join('results', folder_name)
job_directory = os.path.join(folder_directory, 'job_files')
os.makedirs(job_directory, exist_ok=True)

for Gamma in Gamma_list:
    job_name = f'Gamma_{Gamma:.2f}'
    os.makedirs(os.path.join(folder_directory, job_name), exist_ok=True)
    job_file = os.path.join(job_directory, f'{job_name}.job')
    plot_file = os.path.join(folder_directory, job_name, 'plot')
    save_file = os.path.join(folder_directory, job_name, 'data')

    with open(job_file, 'w') as f:
        f.writelines("#!/bin/bash\n")
        f.writelines("#\n")
        f.writelines(f"#SBATCH --job-name={job_name}\n")
        f.writelines(f"#SBATCH --output={os.path.join(job_directory, job_name)}.out\n")
        f.writelines(f"#SBATCH --error={os.path.join(job_directory, job_name)}.err\n")
        f.writelines("#SBATCH --ntasks=1\n")
        f.writelines("#\n")
        f.writelines(
            f"python spin_chain_1d.py --Gamma {Gamma} --Omega {Omega} --E {E} --mu {mu} "
            f"--tmax {tmax} --nt {nt} --T {' '.join(str(T) for T in T_list)} --n-omega {n_omega} "
            f"--measure {' '.join(measure)} "
            f"--plot-file {plot_file} --save {save_file}\n"
        )
    
    os.system(f"sbatch {job_file}")
    