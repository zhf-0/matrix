#!/usr/bin/env python3
# 
import numpy as np
from scipy.sparse import bmat

from fealpy.decorator import cartesian
from fealpy.mesh import MeshFactory as MF
from fealpy.mesh.HalfEdgeMesh2d import HalfEdgeMesh2d
from fealpy.mesh import PolygonMesh
from fealpy.functionspace import ScaledMonomialSpace2d
from .Parameters import Parameter

# class SqrtData():
class PDE():
    """
        \Delta^2 u - \Delta u = f

        u = (x^2 + y^2)^k

    """
    def __init__(self,x0,x1,y0,y1, k = 3):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.k = k

    def domain(self):
        return np.array([self.x0, self.x1,self.y0, self.y1])

    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        k = self.k
        z = x**2+y**2
        return z**k

    @cartesian
    def source(self, p):
        """the right hand side of Possion equation
        INPUT:
            p: array object, N*2
        """
        x = p[..., 0]
        y = p[..., 1]
        z = x**2+y**2
        k = self.k
        F = 16*(k**2)*((k-1)**2)*(z**(k-2))-4*(k**2)*(z**(k-1))
        return F

    @cartesian
    def gradient(self, p):
        """ The gradient of the exact solution
        """
        x = p[..., 0]
        y = p[..., 1]
        z = x**2+y**2
        k = self.k
        val = np.zeros(p.shape, dtype=p.dtype)
        val[..., 0] = 2*k*x*z**(k-1)
        val[..., 1] = 2*k*y*z**(k-1)
        return val

    @cartesian
    def dirichlet(self, p):
        """Dilichlet boundary condition
        """
        return self.solution(p)

def GenerateMat(nx,ny, space_p=1, alpha=1, beta=1):
    x0 = 0.0
    x1 = 1.0
    y0 = 0.0
    y1 = 1.0
    pde = PDE(x0,x1,y0,y1)
    domain = pde.domain()
    mesh = MF.boxmesh2d(domain, nx=nx, ny=ny, meshtype='tri')
    mesh = HalfEdgeMesh2d.from_mesh(mesh)
    mesh.ds.NV = 3
    mesh.init_level_info()
    mesh = PolygonMesh.from_mesh(mesh)

    space = ScaledMonomialSpace2d(mesh,space_p)
    
    isInEdge = ~mesh.ds.boundary_edge_flag()
    isBdEdge = mesh.ds.boundary_edge_flag()

    #组装矩阵
    A = space.stiff_matrix()
    J = space.penalty_matrix(index=isInEdge)
    Q = space.normal_grad_penalty_matrix(index=isInEdge)
    S0 = space.flux_matrix(index=isInEdge)
    S1 = space.flux_matrix()

    A11 = A-S0-S1.T+alpha*J+beta*Q
    A12 = -space.mass_matrix()
    A22 = A11.T-A12
    A21 = alpha*space.penalty_matrix() 
    AD = bmat([[A11, A12], [A21, A22]], format='csr')

    #组装右端向量
    F11 = space.edge_source_vector(pde.gradient, index=isBdEdge, hpower=0)
    F12 = -space.edge_normal_source_vector(pde.dirichlet, index=isBdEdge)
    F21 = space.edge_source_vector(pde.dirichlet, index=isBdEdge)
    F22 = space.source_vector0(pde.source)
    F = np.r_[F11+F12, F21+F22]

    eps = 10**(-15)
    AD.data[ np.abs(AD.data) < eps ] = 0
    AD.eliminate_zeros()
    return AD

class Para(Parameter):
    def __init__(self):
        super().__init__()
        self.DefineRandFloat('alpha',0.1,10)
        self.DefineRandFloat('beta',0.1,10)

        self.DefineRandInt('space_p',1,4)

        if self.para['space_p'] == 1:
            self.DefineRandInt('nx', 20, 65)
        elif self.para['space_p'] == 2:
            self.DefineRandInt('nx', 10, 45)
        elif self.para['space_p'] == 3:
            self.DefineRandInt('nx', 10, 35)

        self.CopyValue('nx', 'ny')

if __name__ == '__main__':
    print('p=1')
    a = GenerateMat(65,65,space_p = 1)
    print(a.shape,a.nnz)

    a = GenerateMat(20,20,space_p = 1)
    print(a.shape,a.nnz)

    print('p=2')
    a = GenerateMat(45,45,space_p = 2)
    print(a.shape,a.nnz)
    a = GenerateMat(10,10,space_p = 2)
    print(a.shape,a.nnz)

    print('p=3')
    a = GenerateMat(35,35,space_p = 3)
    print(a.shape,a.nnz)
    a = GenerateMat(10,10,space_p = 3)
    print(a.shape,a.nnz)
