import numpy as np

from scipy.sparse import spdiags, bmat

from scipy.sparse import csr_matrix,hstack,vstack,spdiags,bmat
import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import ScaledMonomialSpace2d 
from fealpy.boundarycondition import DirichletBC 

from fealpy.decorator import cartesian
from .Parameters import Parameter


# class StokesModelData_5:
class PDE:
    '''
    \mu \Delta u - \nabla p + f = 0
                 \nabla \cdot u = 0
    '''
    def __init__(self,x0,x1,y0,y1, nu):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.nu = nu

    def domain(self):
        return [self.x0, self.x1, self.y0, self.y1]

    @cartesian
    def velocity(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)
        val[..., 0] = 2*(x**3 - x)**2*(3*y**2 - 1)*(y**3 - y) 
        val[..., 1] = (3*x**2 - 1)*(-2*x**3 + 2*x)*(y**3 - y)**2 
        return val

    @cartesian
    def pressure(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 1/(x**2 + 1) - pi/4
        return val

    @cartesian
    def strain(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape + (2, ), dtype=np.float)
        val[..., 0, 0] = 2*(6*x**2 - 2)*(x**3 - x)*(3*y**2 - 1)*(y**3 - y)   
        val[..., 0, 1] += 3*x*(-2*x**3 + 2*x)*(y**3 - y)**2 
        val[..., 0, 1] += 6*y*(x**3 - x)**2*(y**3 - y) 
        val[..., 0, 1] += (2 - 6*x**2)*(3*x**2 - 1)*(y**3 - y)**2/2 
        val[..., 0, 1] += (x**3 - x)**2*(3*y**2 - 1)**2 
        val[..., 1, 0] = val[..., 0, 1] 
        val[..., 1, 1] = (3*x**2 - 1)*(-2*x**3 + 2*x)*(6*y**2 - 2)*(y**3 - y)  
        return val

    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        val = np.zeros(p.shape, dtype=np.float)

        val[..., 0] -= 3*x*(-2*x**3 + 2*x)*(6*y**2 - 2)*(y**3 - y) 
        val[..., 0] -= 12*x*(x**3 - x)*(6*y**2 - 2)*(y**3 - y) 
        val[..., 0] += 2*x/(x**2 + 1)**2 
        val[..., 0] -= 18*y*(x**3 - x)**2*(3*y**2 - 1) 
        val[..., 0] -= (2 - 6*x**2)*(3*x**2 - 1)*(6*y**2 - 2)*(y**3 - y)/2 
        val[..., 0] -= (3*x**2 - 1)*(6*x**2 - 2)*(6*y**2 - 2)*(y**3 - y) 
        val[..., 0] -= 6*(x**3 - x)**2*(y**3 - y) 

        val[..., 1] -= 6*x*(2 - 6*x**2)*(y**3 - y)**2 
        val[..., 1] += 6*x*(3*x**2 - 1)*(y**3 - y)**2 
        val[..., 1] -= 12*y*(3*x**2 - 1)*(-2*x**3 + 2*x)*(y**3 - y) 
        val[..., 1] -= 6*y*(6*x**2 - 2)*(x**3 - x)*(y**3 - y) 
        val[..., 1] -= (3*x**2 - 1)*(-2*x**3 + 2*x)*(3*y**2 - 1)*(6*y**2 - 2) 
        val[..., 1] -= (6*x**2 - 2)*(x**3 - x)*(3*y**2 - 1)**2 
        val[..., 1] -= 3*(-2*x**3 + 2*x)*(y**3 - y)**2
        return val

    @cartesian
    def dirichlet(self, p):
        return self.velocity(p)

def GenerateMat(nx,ny, space_p = 2, nu=1):
    # space_p must >= 2
    assert space_p >= 2

    x0 = 0.0
    x1 = 1.0
    y0 = 0.0
    y1 = 1.0
    
    pde = PDE(x0,x1,y0,y1,nu)
    domain = pde.domain()

    mesh = MF.boxmesh2d(domain, nx=nx, ny=ny, meshtype='tri')
    uspace = LagrangeFiniteElementSpace(mesh, p=space_p)
    pspace = LagrangeFiniteElementSpace(mesh, p=space_p-1)
        
    ugdof = uspace.number_of_global_dofs()
    pgdof = pspace.number_of_global_dofs()

    A = 1/2*uspace.stiff_matrix()
    B0, B1 = uspace.div_matrix(pspace)
    F = uspace.source_vector(pde.source, dim=2)    
    
    qf = mesh.integrator(4,'cell')
    bcs,ws = qf.get_quadrature_points_and_weights()

    
    AA = bmat([[A, None,B0], [None, A, B1], [B0.T, B1.T, None]], format='csr')
    FF = np.r_['0', F.T.flat, np.zeros(pgdof)]
    
    isBdDof = uspace.is_boundary_dof()
    gdof = 2*ugdof + pgdof
    x = np.zeros(gdof,np.float)
    ipoint = uspace.interpolation_points()
    uso = pde.dirichlet(ipoint)
    x[0:ugdof][isBdDof] = uso[:,0][isBdDof]
    x[ugdof:2*ugdof][isBdDof] = uso[isBdDof][:,1]
   
    isBdDof = np.block([isBdDof, isBdDof, np.zeros(pgdof, dtype=np.bool)])

    FF -= AA@x
    bdIdx = np.zeros(gdof, dtype=np.int_)
    bdIdx[isBdDof] = 1
    Tbd = spdiags(bdIdx, 0, gdof, gdof)
    T = spdiags(1-bdIdx, 0, gdof, gdof)
    AA = T@AA@T + Tbd
    FF[isBdDof] = x[isBdDof]

    eps = 10**(-15)
    AA.data[ np.abs(AA.data) < eps ] = 0
    AA.eliminate_zeros()
    return AA

class Para(Parameter):
    def __init__(self):
        super().__init__()

        self.DefineRandInt('space_p',2,4)

        if self.para['space_p'] == 2:
            self.DefineRandInt('nx', 30, 75)
        elif self.para['space_p'] == 3:
            self.DefineRandInt('nx', 20, 50)

        self.CopyValue('nx', 'ny')


if __name__ == '__main__':
    print('mesh is tri, p=2')
    a = GenerateMat(75,75,space_p=2)
    print(a.shape,a.nnz)

    a = GenerateMat(30,30,space_p=2)
    print(a.shape,a.nnz)

    print('mesh is tri, p=3')
    a = GenerateMat(50,50,space_p=3)
    print(a.shape,a.nnz)

    a = GenerateMat(20,20,space_p=3)
    print(a.shape,a.nnz)

