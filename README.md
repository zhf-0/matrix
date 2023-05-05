# Open Matrix Dataset Generator

Generate dataset including matrices and labels for deep learning. 

The architecture of Open Matrix Dataset Generator (OMDG) is:

![image](./doc/pic/arch.png)

OMDG can generate matrices from different programs and solve the corresponding linear equations with different iterative methods simultaneously. Other features include:

- Multi-task parallel
- Breakpoint resume
- Customized labeling method
- Reproduce matrices and right hand vectors from existed configuration file
- Support discretization programs and solvers that written by different languages (C/C++, python, etc. )
- Provide query and download interfaces for SuiteSparse Matrix Collection

**Those features are verified in PC and cluster**. However, the shortcomings of OMDG are:

- The discretization programs and solvers can be written by `MPI`, but can only be executed sequentially, which means can not be executed by `mpirun ...` 
- There will be memory/cache competition if too many tasks running simultaneously, and  increase the wall time

## Installation

- `git clone ...` or download the tar file
- Optional:
  - `FEALPY`: discretization software, [installation](https://github.com/weihuayi/fealpy) 
  - `petsc4py`: solver, [installation](https://www.mcs.anl.gov/petsc/petsc4py-current/docs/usrman/install.html)
  - `ssgetpy`: download matrices from`SuiteSpare Matrix Collection`, [installation](https://github.com/drdarshan/ssgetpy) 

 