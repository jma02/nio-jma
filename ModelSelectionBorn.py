import itertools
import os
import sys
import subprocess

import time
max_parallel_jobs = 2
running_processes = []

import numpy as np

np.random.seed(3545)

model = "nio"
which = "born_farfield"

random = True
cluster = "false"
sbatch = False
ablation_fno = False
cpus = 1
max_workers = 2
script = "RunNio.py"

training_properties_ = {
        "step_size": [15],
        "gamma": [1, 0.98],
        "epochs": [100],
        "batch_size": [256],
        "learning_rate": [0.1, 0.01, 0.001],
        # "norm": ["none", "norm", "minmax", "norm-inp"],
        "norm": ["minmax"],
        "weight_decay": [0., 1e-6],
        "reg_param": [0.],
        "reg_exponent": [1,2],
        "inputs": [2],
        "b_scale": [0.],
        "retrain": [888],
        "mapping_size_ff": [32],
        "scheduler": ["step"]
    }
branch_architecture_ = {
    "n_hidden_layers_b": [3],
    "neurons_b": [64],
    "act_string_b": ["leaky_relu"],
    "dropout_rate_b": [0.0],
    "kernel_size_b": [3],
}

trunk_architecture_ = {
    "n_hidden_layers_t": [4, 6, 8],
    "neurons_t": [256],
    "act_string_t": ["leaky_relu"],
    "dropout_rate_t": [0.0],
    "n_basis_t": [50,100]
}

fno_architecture_ = {
    "width": [32, 64],
    "modes": [16, 32],
    "n_layers": [2, 3, 4]
}

denseblock_architecture_ = {
    "n_hidden_layers_db": [2],
    "neurons_db": [50],
    "act_string_db": ["leaky_relu"],
    "retrain_db": [127],
    "dropout_rate_db": [0.0],
}
if ablation_fno:
    folder_name = "ModelSelection_final_s_abl_" + which + "_" + model
else:
    folder_name = "ModelSelection_final_s_" + which + "_" + model

ndic = {**training_properties_,
        **branch_architecture_,
        **trunk_architecture_,
        **fno_architecture_,
        **denseblock_architecture_}
if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
settings = list(itertools.product(*ndic.values()))

i = 0
nconf = 30
if len(settings) < nconf:
    nconf = len(settings)

if random:
    idx = np.random.choice(len(settings), nconf, replace=False)
    settings = np.array(settings)[idx].tolist()


for setup in settings:
    # time.sleep(10)
    # print(setup)

    folder_path = "\'" + folder_name + "/Setup_" + str(i) + "\'"
    print("###################################")
    training_properties_ = {
        "step_size": int(setup[0]),
        "gamma": float(setup[1]),
        "epochs": int(setup[2]),
        "batch_size": int(setup[3]),
        "learning_rate": float(setup[4]),
        "norm": setup[5],
        "weight_decay": float(setup[6]),
        "reg_param": float(setup[7]),
        "reg_exponent": int(setup[8]),
        "inputs": int(setup[9]),
        "b_scale": float(setup[10]),
        "retrain": int(setup[11]),
        "mapping_size_ff": int(setup[12]),
        "scheduler": setup[13]
    }

    branch_architecture_ = {
        "n_hidden_layers": int(setup[14]),
        "neurons": int(setup[15]),
        "act_string": setup[16],
        "dropout_rate": float(setup[17]),
        "kernel_size": int(setup[18])
    }

    trunk_architecture_ = {
        "n_hidden_layers": int(setup[19]),
        "neurons": int(setup[20]),
        "act_string": setup[21],
        "dropout_rate": float(setup[22]),
        "n_basis": int(setup[23])
    }

    fno_architecture_ = {
        "width": int(setup[24]),
        "modes": int(setup[25]),
        "n_layers": int(setup[26])
    }

    denseblock_architecture_ = {
        "n_hidden_layers": int(setup[27]),
        "neurons": int(setup[28]),
        "act_string": setup[29],
        "retrain": int(setup[30]),
        "dropout_rate": float(setup[31])
    }
    arguments = list()
    arguments.append(folder_path)
    if sbatch:
        arguments.append("\\\"" + str(training_properties_) + "\\\"")
    else:
        arguments.append("\'" + str(training_properties_).replace("\'", "\"") + "\'")


    if sbatch:
        arguments.append("\\\"" + str(branch_architecture_) + "\\\"")
    else:
        arguments.append("\'" + str(branch_architecture_).replace("\'", "\"") + "\'")


    #
    if sbatch:
        arguments.append("\\\"" + str(trunk_architecture_) + "\\\"")
    else:
        arguments.append("\'" + str(trunk_architecture_).replace("\'", "\"") + "\'")

    if sbatch:
        arguments.append("\\\"" + str(fno_architecture_) + "\\\"")
    else:
        arguments.append("\'" + str(fno_architecture_).replace("\'", "\"") + "\'")

    if sbatch:
        arguments.append("\\\"" + str(denseblock_architecture_) + "\\\"")
    else:
        arguments.append("\'" + str(denseblock_architecture_).replace("\'", "\"") + "\'")


    arguments.append(which)
    arguments.append(model)
    arguments.append(max_workers)
    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if cluster == "true":
            # string_to_exec = "bsub -W 16:00 -n 32 -R \'rusage[mem=2048]\' -R \'rusage[ngpus_excl_p=1]\' python3 OpSquareBigData.py"
            # string_to_exec = "bsub -W 16:00 -n 32 -R \'rusage[mem=2048]\' -R \'rusage[ngpus_excl_p=1]\' python3 CNOFWI.py"
            # string_to_exec = "bsub -W 16:00 -n 32 -R \'rusage[mem=2048]\' -R \'rusage[ngpus_excl_p=1]\' python3 OpSquarePoissonNew.py"
            # string_to_exec = "bsub -W 16:00 -n 32 -R \'rusage[mem=2048]\' -R \'rusage[ngpus_excl_p=1]\' python3 OpSquareRad.py"
            if sbatch:
                if which == "curve" or which == "style":
                    string_to_exec = "sbatch --time=72:00:00 -n " + str(cpus) + " -G 1 --mem-per-cpu=16384 --wrap=\" python3 " + script + " "
                else:
                    string_to_exec = "sbatch --time=24:00:00 -n " + str(cpus) + " -G 1 --mem-per-cpu=16384 --wrap=\" python3 " + script + " "
        else:
            string_to_exec = "python3 " + script + " "
        for arg in arguments:
            string_to_exec = string_to_exec + " " + str(arg)
        if cluster and sbatch:
            string_to_exec = string_to_exec + " \""
        print(string_to_exec)
           # Wait for a slot to be available
        while len(running_processes) >= max_parallel_jobs:
            # Check if any process has finished
            for j, process in enumerate(running_processes):
                if process.poll() is not None:  # Process has finished
                    print(f"Job finished with return code: {process.returncode}")
                    running_processes.pop(j)
                    break
            else:
                # No process finished, wait a bit
                time.sleep(1)
         # Set up environment for GPU assignment
        gpu_id = i % max_parallel_jobs
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Create log files for output
        log_dir = folder_name + "/Setup_" + str(i)
        if not os.path.exists(log_dir.strip("'")):
            os.makedirs(log_dir.strip("'"))

        # With this:
        log_dir_clean = log_dir.strip("'")
        stdout_file = open(f"{log_dir_clean}/stdout.log", 'w')
        stderr_file = open(f"{log_dir_clean}/stderr.log", 'w')
        
        print(f"Starting job {i} on GPU {gpu_id}")
        
        # Start the process
        process = subprocess.Popen(
            string_to_exec,
            shell=True,  # Let shell handle argument parsing
            env=env,
            stdout=stdout_file,
            stderr=stderr_file,
            preexec_fn=os.setsid  # Create new process group
        )
        
        # Store process info
        process.stdout_file = stdout_file
        process.stderr_file = stderr_file
        process.job_id = i
        process.gpu_id = gpu_id
        
        running_processes.append(process)
        
        

    i = i + 1
# Wait for all remaining processes to finish
print("Waiting for all remaining jobs to complete...")
while running_processes:
    for j, process in enumerate(running_processes):
        if process.poll() is not None:  # Process has finished
            print(f"Job {process.job_id} on GPU {process.gpu_id} finished with return code: {process.returncode}")
            process.stdout_file.close()
            process.stderr_file.close()
            running_processes.pop(j)
            break
    else:
        time.sleep(1)

print("All jobs completed!")