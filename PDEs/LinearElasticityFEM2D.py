#!/usr/bin/env python3
# 
import numpy as np

from fealpy.mesh import MeshFactory as MF
from fealpy.decorator import cartesian
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC, NeumannBC
from .Parameters import Parameter
from .Utility import WriteMatAndVec



class PDE():
    def __init__(self,x0,x1,y0,y1, E, nu):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

        self.E = E 
        self.nu = nu
        self.lam = self.nu*self.E/((1+self.nu)*(1-2*self.nu))
        self.mu = self.E/(2*(1+self.nu))

    def domain(self):
        return [self.x0, self.x1, self.y0, self.y1]
    
    @cartesian
    def displacement(self, p):
        return 0.0

    @cartesian
    def jacobian(self, p):
        return 0.0

    @cartesian
    def strain(self, p):
        return 0.0

    @cartesian
    def stress(self, p):
        return 0.0

    @cartesian
    def source(self, p):
        val = np.array([0.0, 0.0], dtype=np.float64)
        shape = len(p.shape[:-1])*(1, ) + (2, )
        return val.reshape(shape)

    @cartesian
    def dirichlet(self, p):
        val = np.array([0.0, 0.0], dtype=np.float64)
        shape = len(p.shape[:-1])*(1, ) + (2, )
        return val.reshape(shape)

    @cartesian
    def neumann(self, p, n):
        val = np.array([-500, 0.0], dtype=np.float64)
        shape = len(p.shape[:-1])*(1, ) + (2, )
        return val.reshape(shape)

    @cartesian
    def is_dirichlet_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = np.abs(x) < 1e-13
        return flag

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = np.abs(x - 1) < 1e-13
        return flag


def GenerateMat(nx,ny, mat_type=None, mat_path=None, need_rhs=False, 
                space_p=1, E=1e+5, nu=0.2):
    x0 = 0.0
    x1 = 1.0
    y0 = 0.0
    y1 = 1.0
    pde = PDE(x0,x1,y0,y1,E,nu)
    domain = pde.domain()
    mesh = MF.boxmesh2d(domain, nx=nx, ny=ny, meshtype='tri')
    space = LagrangeFiniteElementSpace(mesh, p=space_p)

    bc0 = DirichletBC(space, pde.dirichlet, threshold=pde.is_dirichlet_boundary) 
    bc1 = NeumannBC(space, pde.neumann, threshold=pde.is_neumann_boundary)

    uh = space.function(dim=2) # (gdof, 2) and vector fem function uh[i, j] 
    A = space.linear_elasticity_matrix(pde.lam, pde.mu) # (2*gdof, 2*gdof)
    F = space.source_vector(pde.source, dim=2) 
    F = bc1.apply(F)
    A, F = bc0.apply(A, F, uh)

    eps = 10**(-15)
    A.data[ np.abs(A.data) < eps ] = 0
    A.eliminate_zeros()
    
    ########################################################
    #            write matrix A and rhs vector F           #
    ########################################################
    WriteMatAndVec(A,F,mat_type,mat_path,need_rhs)

    row_num, col_num = A.shape
    nnz = A.nnz
    return row_num, col_num, nnz

class Para(Parameter):
    def __init__(self):
        super().__init__()
        
    def AddParas(self):
        self.DefineRandFloat('E',1e4,5e5)
        self.DefineRandFloat('nu',0.01,1.0)

        self.DefineRandInt('space_p',1,4)

        if self.para['space_p'] == 1:
            self.DefineRandInt('nx', 80, 160)
        elif self.para['space_p'] == 2:
            self.DefineRandInt('nx', 40, 80)
        elif self.para['space_p'] == 3:
            self.DefineRandInt('nx', 30, 50)

        self.CopyValue('nx', 'ny')


if __name__ == '__main__':
    print('p=1')
    row_num, col_num, nnz = GenerateMat(160,160,space_p = 1)
    print(row_num, col_num, nnz)
    row_num, col_num, nnz = GenerateMat(80,80,space_p = 1)
    print(row_num, col_num, nnz)

    print('p=2')
    row_num, col_num, nnz = GenerateMat(80,80,space_p = 2)
    print(row_num, col_num, nnz)
    row_num, col_num, nnz = GenerateMat(40,40,space_p = 2)
    print(row_num, col_num, nnz)

    print('p=3')
    row_num, col_num, nnz = GenerateMat(50,50,space_p = 3)
    print(row_num, col_num, nnz)
    row_num, col_num, nnz = GenerateMat(30,30,space_p = 3)
    print(row_num, col_num, nnz)
