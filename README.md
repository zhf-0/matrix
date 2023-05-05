# Open Matrix Dataset Generator

This is a sparse matrix generator which is backed by PDE model problems. It generates data sets including matrices and labels mainly for deep learning training purpopses. Apparently, it can be used to generate test problems for sparse linear solvers.

The architecture of Open Matrix Dataset Generator (OMDG) is:

![image](./doc/pic/arch.png)

OMDG can generate matrices from different programs and solve the corresponding linear equations with different iterative methods simultaneously. Other features include:

- Multi-tasking
- Breakpoint and resume
- Customized labeling method
- Reproduce matrices and right hand vectors from existed configuration file
- Support discretization programs and solvers that written by different languages (C/C++, python, etc. )
- Provide query and download interfaces for SuiteSparse Matrix Collection

**Those features are verified in PC and cluster**. However, the shortcomings of OMDG are:

- The discretization programs and solvers can be written by `MPI`, but can only be executed sequentially, which means can not be executed by `mpirun ...` 
- There will be memory/cache competition if too many tasks running simultaneously, and  increase the wall time

# Installation

- `git clone git@github.com:OpenCAXPlus/OMDG.git` or download the tar file directly
- Optional:
  - `FEALPY`: discretization software, [installation](https://github.com/weihuayi/fealpy) 
  - `petsc4py`: solver, [installation](https://www.mcs.anl.gov/petsc/petsc4py-current/docs/usrman/install.html)
  - `ssgetpy`: download matrices from`SuiteSpare Matrix Collection`, [installation](https://github.com/drdarshan/ssgetpy) 

# Usage

Using the matrix from Poisson equation as the example.

- First step is generating 

```python
import PDEs.PoissonFEM2d as pde1

# index of the matrix
idx = 1

# the type of output matrix, SciCSR: scipy.sparse.csr_matrix(default) 
#                            SciCOO: scipy.sparse.coo_matrix 
#                            COO: coo in txt format
mat_type = 'SciCSR'

# matrix type, the name has to be choosen from the following list:
# if the format is 'SciCSR', the name is 'scipy_csr{index}.npz'
# if the format is 'SciCOO', the name is 'scipy_coo{index}.npz'
# if the format is 'COO', the name is 'coo{index}.npz'
mat_path = f'scipy_csr{idx}.npz'

# don't need to output the right hand side vector 
need_rhs = 0

# pre-defined parameters in Poisson equation
parameter = pde1.Para()

# register new parameters
parameter.DefineFixPara('mat_type',mat_type)
parameter.DefineFixPara('mat_path',mat_path)
parameter.DefineFixPara('need_rhs',need_rhs)
parameter.DefineFixPara('seed',0)

# generate matrix 
row_num, col_num, nnz = pde1.GenerateMat(**parameter.para)

# add infomation, `row_num`, `col_num`, `nnz` can be any number 
parameter.others['PDE_type'] = 1
parameter.others['row_num'] = row_num
parameter.others['col_num'] = col_num
parameter.others['nnz'] = nnz
```

- Second step is solving

```python
import PetscSolvers

# each iterative method will solve 3 times for average
batch_size = 3 

# where to store all json files that include the config and result infomation
json_dir = './JsonFiles/' 

# define the solver, 
# `summary.json` is the file that includes statistic infomation
# `num_cpu` is the number of task executed simultaneously, one task per cpu  
solver = PetscSolvers.ParaSolveAndAnalysis(
         json_dir,
         batch_size,
         'summary.json',
         num_cpu=8)

# solve the matrix with 28 iterative methods in PETSc
# finish solving, the results and configs information are in  json_dir/result{idx}.json
solver.Process(idx,mat_path,parameter,need_rhs)
```

