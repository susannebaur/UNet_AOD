#!/bin/bash
  
# Instructions SBATCH always at the beginning of the script!

# Change the working directory before the execution of the job.
# Warning: the environment variables, e.g. $HOME,
# are not interpreted for the SBATCH instructions.
# Writing absolute paths is recommended.
#SBATCH --chdir=/home/USERNAME/jobs/   # <-- Replace USERNAME with your own username

# The job partition (maximum elapsed time of the job).
#SBATCH --partition=batch

# The name of the job.
#SBATCH --job-name=data_prep

# The number of GPU cards requested.
# If the GPU architecture is not specified,
# Slurm chooses first Ada then Ampere then Turing.
#SBATCH --gpus=1

# The maximum elapsed time of computation, expressed in hour.
# Default is 1 hour.
#SBATCH --time=48:00:00

# Email notifications (e.g., the beginning and the end of the job).
#SBATCH --mail-user=susanne.baur@meteo.fr
#SBATCH --mail-type=all

# The path of the job log files.
# The error and the output logs can be merged into the same file.
# %j is the job id.
#SBATCH --error=/home/sbaur/jobs/job-data_prep-%j.log
#SBATCH --output=/home/sbaur/jobs/job-data_prep-%j.log

# Overtakes the system memory limits.
ulimit -s unlimited

# Load the user configuration file.
source /etc/profile

# Unload all modules previously loaded.
#module purge

# Changes to the given working directory.
# It supersedes the #SBATCH --chdir instruction,
# and environment variables can be used (e.g., ${HOME}).
#cd ${HOME}//python_scripts/MRV-project/NN_model/UNet/

# Load the required module.
module load 'pytorch/2.6.0' # Or any other AI module.

################ OPTIONAL ENVIRONMENT ##################

##### PYTHON VIRTUAL ENVIRONMENT ACTIVATION
## the extended Python module should be loaded before.
# source "path/to/myenv/bin/activate" 

########################################################

# Enable the standard and error outputs of Python.
export PYTHONUNBUFFERED=1

####################### DEBUG #####################
# Optional instructions.
cat << EOF
===== my job information ====
> module list:
`module list 2>&1`
> node list: ${SLURM_NODELIST}
> my job id: ${SLURM_JOB_ID}
> job name: ${SLURM_JOB_NAME}
> partition: ${SLURM_JOB_PARTITION}
> submit host: ${SLURM_SUBMIT_HOST}
> submit directory: ${SLURM_SUBMIT_DIR}
> current directory: `pwd`
> executed as user: `whoami`
> executed as slurm user: ${SLURM_JOB_USER}
> user PATH: ${PATH}
EOF
####################################################

# Run your process. The optional command line arguments of the job.slurm script
# are passed to your Python scripts ($@).
#python mycode.py $@ 
cd /spiritx-home/sbaur/python_scripts/MRV-project/NN_model/UNet/
./prepro_data.sh


returned_code=$?
echo "> script completed with exit code ${returned_code}"
exit ${returned_code}
