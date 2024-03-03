import os
import sys
sys.path.append('./Solve/')
from Solver import ParTaskParRunCluster

def TestParTaskParRun():

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

    script_file = 'solve.sh'
    header = ['#!/bin/bash \n']
    footer = ['echo finished !! \n']
    a.GenerateScript(script_file,header,footer,command)

if __name__ == '__main__':
    TestParTaskParRun()
