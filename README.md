# OpenMat

This is a sparse matrix generator which is backed by PDE model problems. It generates data sets including matrices and labels mainly for deep learning training purpopses. Apparently, it can be used to generate test problems for sparse linear solvers.

The architecture of OpenMat is:

![image](./doc/pic/arch.png)

OpenMat can generate matrices from different programs and solve the corresponding linear equations with different iterative methods simultaneously. Other features include:

- Multi-tasking
- Breakpoint and resume
-  Label computation
- Reproduce matrices and right hand vectors from existed configuration file
- Support discretization programs and solvers that written by different languages (`C/C++`, `python`, etc. )
- Provide query and download interfaces for SuiteSparse Matrix Collection

**Above features are verified in PC and cluster**. 



# Installation

- `git clone git@github.com:zhf-0/matrix.git` or download the tar file directly
- Optional:
  - `FEALPY`: discretization software, [installation](https://github.com/weihuayi/fealpy) 
  - `petsc4py`: solver, [installation](https://www.mcs.anl.gov/petsc/petsc4py-current/docs/usrman/install.html)
  - `ssgetpy`: download matrices from`SuiteSpare Matrix Collection`, [installation](https://github.com/drdarshan/ssgetpy) 

# Usage

Using the matrix from Poisson equation as the example.

- First step is generating 

```python
# index of the matrix
idx = 1

# the type of output matrix, SciCSR: scipy.sparse.csr_matrix(default) 
#                            SciCOO: scipy.sparse.coo_matrix 
#                            COO: coo in txt format
mat_type = 'SciCSR'

# matrix name, the name has to be choosen from the following list:
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
# each iterative method will solve 3 times for average
batch_size = 3 

# where to store all json files that include the config and result infomation
json_dir = './JsonFiles/' 

```

