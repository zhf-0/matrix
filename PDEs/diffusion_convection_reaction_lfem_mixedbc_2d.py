import random
import argparse
import numpy as np

from fealpy.mesh import TriangleMesh, QuadrangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import ScalarDiffusionIntegrator      # (A\nabla u, \nabla v) 
from fealpy.fem import ScalarConvectionIntegrator     # (b\cdot \nabla u, v) 
from fealpy.fem import ScalarMassIntegrator           # (r*u, v)
from fealpy.fem import ScalarSourceIntegrator         # (f, v)
from fealpy.fem import ScalarNeumannSourceIntegrator  # <g_N, v>
from fealpy.fem import ScalarRobinSourceIntegrator    # <g_R, v>
from fealpy.fem import ScalarRobinBoundaryIntegrator  # <kappa*u, v>
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import DirichletBC
from fealpy.decorator import cartesian

from Parameters import Parameter
from Utility import WriteMatAndVec

class PDE:
    """
	Equation:
        -\\nabla\cdot(A(x)\\nabla u + b(x)u) + cu = f in \Omega
	
	B.C.:
	u = g_D on \partial\Omega
	
	Using fx = cos(pi*x)*cos(pi*y) to construct boundary condition,
    which is not the solution of the PDE
	
	Coefficients:
	A(x) = [10.0, -1.0; -1.0, 2.0]
	b(x) = [-1; -1]
	c(x) = 1 + x^2 + y^2
    """
    def __init__(self,x0,x1,y0,y1,blockx,blocky,kappa=1.0):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.xstep = (x1-x0)/blockx 
        self.ystep = (y1-y0)/blocky
        self.coef1 = 10**np.random.uniform(0.0,5.0,(blocky+1,blockx+1))
        self.coef2 = 10**np.random.uniform(0.0,5.0,(blocky+1,blockx+1))

        # only use in robin boundary condition
        self.kappa = kappa

    def domain(self):
        return np.array([self.x0, self.x1,self.y0, self.y1])
    
    
    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 12*pi*pi*np.cos(pi*x)*np.cos(pi*y) 
        val += 2*pi*pi*np.sin(pi*x)*np.sin(pi*y) 
        val += np.cos(pi*x)*np.cos(pi*y)*(x**2 + y**2 + 1) 
        val -= pi*np.cos(pi*x)*np.sin(pi*y) 
        val -= pi*np.cos(pi*y)*np.sin(pi*x)
        return val

    @cartesian
    def fx(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val # val.shape == x.shape

    @cartesian
    def grad_fx(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val # val.shape == p.shape

    # @cartesian
    # def diffusion_coefficient(self, p):
    #     return np.array([[10.0, -1.0], [-1.0, 2.0]], dtype=np.float64)

    @cartesian
    def diffusion_coefficient(self, p):
        '''
        the coefficients are varying based on the coordinates
        '''
        x = p[..., 0]
        y = p[..., 1]
        xidx = x//self.xstep
        xidx = xidx.astype(np.int)
        yidx = y//self.ystep 
        yidx = yidx.astype(np.int)

        shape = p.shape+(2,)
        val = np.zeros(shape,dtype=np.float64)
        val[...,0,0] = self.coef1[xidx,yidx]
        val[...,0,1] = 0.0

        val[...,1,0] = 0.0
        val[...,1,1] = self.coef2[xidx,yidx]
        return val

    @cartesian
    def convection_coefficient(self, p):
        return np.array([-1.0, -1.0], dtype=np.float64)

    @cartesian
    def reaction_coefficient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        return 1 + x**2 + y**2

    @cartesian
    def dirichlet(self, p):
        return self.fx(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        y = p[..., 1]
        return (np.abs(y - self.y1) < 1e-12) | (np.abs( y -  self.y0) < 1e-12)

    @cartesian
    def neumann(self, p, n):
        grad = self.grad_fx(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        return np.abs(x - self.x1) < 1e-12

    @cartesian
    def robin(self, p, n):
        grad = self.grad_fx(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        val += self.kappa*self.fx(p) 
        return val

    @cartesian
    def is_robin_boundary(self, p):
        x = p[..., 0]
        return np.abs(x - self.x0) < 1e-12


def GenerateMat(nx,ny,blockx,blocky, mat_type=None, mat_path=None, need_rhs=False,
                seed=0, mesh_type='quad', space_p=1):

    np.random.seed(seed)
    random.seed(seed)
    x0 = 0.0
    x1 = 1.0
    y0 = 0.0
    y1 = 1.0
    pde = PDE(x0,x1,y0,y1,blockx,blocky)
    domain = pde.domain()

    if mesh_type == 'tri':
        mesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)
    elif mesh_type == 'quad':
        mesh = QuadrangleMesh.from_box(domain, nx=nx, ny=ny)

    space = LagrangeFESpace(mesh, p=space_p)
    bform = BilinearForm(space)
    # (A(x)\nabla u, \nabla v)
    D = ScalarDiffusionIntegrator(q=space_p+3)
    # (b\cdot \nabla u, v)
    C = ScalarConvectionIntegrator(c=pde.convection_coefficient, q=space_p+3)
    # (r*u, v)
    M = ScalarMassIntegrator(q=space_p+3)
    # <kappa*u, v>
    bform.add_domain_integrator([D, C, M]) 

    R = ScalarRobinBoundaryIntegrator(pde.kappa, threshold=pde.is_robin_boundary, q=space_p+2)
    bform.add_boundary_integrator(R) 
    A = bform.assembly()

    lform = LinearForm(space)
    # (f, v)
    Vs = ScalarSourceIntegrator(pde.source, q=space_p+2)
    # <g_N, v>
    Vn = ScalarNeumannSourceIntegrator(pde.neumann, threshold=pde.is_neumann_boundary, q=space_p+2)
    # <g_R, v>
    Vr = ScalarRobinSourceIntegrator(pde.robin, threshold=pde.is_robin_boundary, q=space_p+2)
    lform.add_domain_integrator(Vs)
    lform.add_boundary_integrator([Vr, Vn])
    F = lform.assembly()

    # Dirichlet 
    bc = DirichletBC(space, pde.dirichlet, threshold=pde.is_dirichlet_boundary) 
    uh = space.function() 
    A, F = bc.apply(A, F, uh)
    
    # eliminate zeros in the matrix 
    eps = 10**(-15)
    A.data[ np.abs(A.data) < eps ] = 0
    A.eliminate_zeros()
    
    # write matrix A and rhs vector F  
    WriteMatAndVec(A,F,mat_type,mat_path,need_rhs)


class Para(Parameter):
    def __init__(self):
        super().__init__()

    def AddParas(self):
        self.RandChoose('mesh_type',['tri','quad'])
        self.DefineRandInt('space_p',1,4)

        if self.para['space_p'] == 1:
            self.DefineRandInt('nx', 50, 200)
            self.DefineRandInt('blockx',20,40)
        elif self.para['space_p'] == 2:
            self.DefineRandInt('nx', 50, 110)
            self.DefineRandInt('blockx',20,40)
        elif self.para['space_p'] == 3:
            self.DefineRandInt('nx', 40, 70)
            self.DefineRandInt('blockx',20,30)

        self.CopyValue('nx', 'ny')
        self.CopyValue('blockx', 'blocky')

def test():
    print('mesh is quad, p=1')
    GenerateMat(200,200,2,2)

    print('mesh is quad, p=2')
    GenerateMat(110,110,2,2,space_p=2)

    print('mesh is quad, p=3')
    GenerateMat(70,70,2,2,space_p=3)


    print('mesh is tri, p=1')
    GenerateMat(200,200,2,2,mesh_type='tri')
    
    print('mesh is tri, p=2')
    GenerateMat(110,110,2,2,mesh_type='tri',space_p=2)
    
    print('mesh is tri, p=3')
    GenerateMat(70,70,2,2,mesh_type='tri',space_p=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', default='1', type=int, dest='nx')
    parser.add_argument('--ny', default='1', type=int, dest='ny')
    parser.add_argument('--blockx', default='1', type=int, dest='blockx')
    parser.add_argument('--blocky', default='1', type=int, dest='blocky')
    parser.add_argument('--mat_type', default=None, type=str, dest='mat_type')
    parser.add_argument('--mat_path', default=None, type=str, dest='mat_path')
    parser.add_argument('--need_rhs', default=False, type=bool, dest='need_rhs')
    parser.add_argument('--seed', default=0, type=int, dest='seed')
    parser.add_argument('--mesh_type', default='quad', type=str, dest='mesh_type')
    parser.add_argument('--space_p', default=1, type=int, dest='space_p')

    args = parser.parse_args()
    nx = args.nx
    ny = args.ny
    blockx = args.blockx
    blocky = args.blocky
    mat_type = args.mat_type
    mat_path = args.mat_path
    need_rhs = args.need_rhs
    seed = args.seed
    mesh_type = args.mesh_type
    space_p = args.space_p
    GenerateMat(nx,ny,blockx,blocky,mat_type,mat_path,need_rhs,seed,mesh_type,space_p)
