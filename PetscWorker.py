import numpy as np
import scipy.sparse as sparse
from scipy.sparse import csr_matrix, csc_matrix 
import json
import time
import argparse
import os

import petsc4py
from petsc4py import PETSc

def RunPetsc(petsc_A,petsc_b,ksp_name,pc_name):
    length = len(petsc_b.getArray())
    petsc_x = PETSc.Vec().createSeq(length) 
    
    opts = PETSc.Options()
    opts["ksp_type"] = ksp_name
    if pc_name == 'BjIlu0QMD':
        opts["pc_type"] = 'bjacobi'
        opts['sub_pc_factor_mat_ordering_type natural'] = 'qmd'
    elif pc_name == 'BjIlu1QMD':
        opts["pc_type"] = 'bjacobi'
        opts['sub_pc_factor_levels'] = 1
        opts['sub_pc_factor_mat_ordering_type natural'] = 'qmd'
    elif pc_name == 'BjLu':
        opts["pc_type"] = 'bjacobi'
        opts['sub_pc_type'] = 'lu'
    elif pc_name == 'BjGmres':
        opts["pc_type"] = 'bjacobi'
        opts['sub_ksp_type'] = 'gmres'
    elif pc_name == 'asm1':
        opts["pc_type"] = 'asm'
    elif pc_name == 'asm2':
        opts["pc_type"] = 'asm'
        opts['pc_asm_overlap'] = 2
    elif pc_name == 'boomer':
        opts["pc_type"] = 'hypre'
        opts["pc_hypre_type"] = 'boomeramg'
    elif pc_name == 'euclid':
        opts["pc_type"] = 'hypre'
        opts["pc_hypre_type"] = 'euclid'
    elif pc_name == 'parasails':
        opts["pc_type"] = 'hypre'
        opts["pc_hypre_type"] = 'parasails'
    else:
        opts["pc_type"] = pc_name

    opts["ksp_rtol"] = 1.0e-6 
    # opts["ksp_atol"] = 1e-10 
    opts["ksp_max_it"] = 1000 
    # opts["ksp_monitor_true_residual"] = None
    # opts["ksp_view"] = None # List progress of solver
    # opts["no_signal_handler"] = None 
    opts["ksp_norm_type"] = 'unpreconditioned'

    ksp = PETSc.KSP().create() 
    ksp.setOperators(petsc_A)
    ksp.setFromOptions()
    # print ('Solving with:', ksp.getType()) # prints the type of solver

    begin = time.time()
    ksp.solve(petsc_b, petsc_x) 
    end = time.time()

    iter_num = ksp.getIterationNumber()
    reason = ksp.getConvergedReason()
    resi = ksp.getResidualNorm()	
    b_norm = petsc_b.norm()
    rlt_resi = resi/b_norm
    elapsed_time = end-begin
    
    print(f'iter_num = {iter_num}')
    print(f'stop_reason = {reason}')
    print(f'residual_norm = {resi}')
    print(f'b_norm = {b_norm}')
    print(f'relative_resi = {rlt_resi}')
    print(f'elapsed_time = {elapsed_time}')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-idx", type=int, dest='idx')
    parser.add_argument("-path", type=str, dest='path')
    parser.add_argument("-ksp", type=str, dest='ksp_name')
    parser.add_argument("-pc", type=str, dest='pc_name')
    parser.add_argument("-rhs", type=int, dest='need_rhs')
    args = parser.parse_args()

    mat_path = args.path
    ksp_name = args.ksp_name
    pc_name = args.pc_name
    need_rhs = args.need_rhs

    # load scipy coo matrix from file and generate petsc matrix 
    csr_A = sparse.load_npz(mat_path)
    petsc_A = PETSc.Mat().createAIJ(size=csr_A.shape, csr=(csr_A.indptr, csr_A.indices, csr_A.data))

    if need_rhs:
        mat_dir = os.path.dirname(mat_path)
        mat_name = os.path.basename(mat_path)
        if mat_name.endswith('npz'):  
            idx_str = mat_name[9:-4]
            rhs_name = f'rhs{idx_str}.npy'
            rhs_path = os.path.join(mat_dir,rhs_name)
            b = np.load(rhs_path)
        else:
            print('other vector format is not supported, use np.ones as the rhs vector')
            b = np.ones(csr_A.shape[0])
    else:
        # generate petsc vector b with all values equal 1.0  
        b = np.ones(csr_A.shape[0])

    petsc_b = PETSc.Vec().createSeq(len(b)) 
    petsc_b.setValues(range(len(b)), b) 

    # try to solve Ax=b with different ksp and pc combinations
    RunPetsc(petsc_A,petsc_b,ksp_name,pc_name)


if __name__ == '__main__':
    main()
