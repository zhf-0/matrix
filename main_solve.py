import os
import sys
sys.path.append('./Solve/')
from Solver import ParTaskParRunCluster

def TestParTaskParRun():
    import re

    yaml_dir = 'YamlFiles'
    json_dir = 'JsonFiles'
    mat_dir = 'PetscMat'
    batch_size = 3
    num_task = 8

    command = './Solve/rs -ksp_type {} -pc_type {} -mat_file {} -yaml_file {} \n'

    mat_template = './MatData/scipy_csr{}.npz'
    idx_list = list( range(1000))
    tmp_list = list(range(1500,2500))
    idx_list.extend(tmp_list)

    a = ParTaskParRunCluster(json_dir,yaml_dir,mat_dir,batch_size,idx_list,num_task)
    a.Process(mat_template)

    script_file = 'run.sh'
    header = ['#!/bin/bash \n']
    header.append('#SBATCH -J zhf1_para \n')
    header.append('#SBATCH -N 1 \n')
    header.append('#SBATCH -n 1 \n')
    header.append('#SBATCH -c 1 \n')
    header.append('#SBATCH -t 24:00:00 \n')
    header.append('#SBATCH -o output/out3.txt \n')
    header.append('#SBATCH -e output/err3.txt \n')
    header.append('module purge \n')
    header.append('module load mpi/openmpi/4.1.1-gcc9.3.0 \n')
    header.append('source /public1/home/sch1190/zhf/software/miniconda/install/bin/activate openmat \n')
    footer = ['echo finished !! \n']
    a.GenerateScript(script_file,header,footer,command)

if __name__ == '__main__':
    TestParTaskParRun()
