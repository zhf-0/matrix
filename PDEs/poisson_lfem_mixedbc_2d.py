import argparse
import numpy as np

from fealpy.mesh import TriangleMesh, QuadrangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import ScalarLaplaceIntegrator       
from fealpy.fem import ScalarSourceIntegrator         
from fealpy.fem import ScalarNeumannSourceIntegrator  
from fealpy.fem import ScalarRobinSourceIntegrator    
from fealpy.fem import ScalarRobinBoundaryIntegrator 
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import DirichletBC
from fealpy.decorator import cartesian

from Parameters import Parameter
from Utility import WriteMatAndVec

class PDE:
    """
        -\\Delta u = f
        u = cos(pi*x)*cos(pi*y)
    """
    def __init__(self,x0,x1,y0,y1,kappa=1.0):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

        # coefficient in robin boundary condition
        self.kappa = kappa 

    def domain(self):
        return np.array([self.x0, self.x1,self.y0, self.y1])

    @cartesian
    def solution(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
        return val # val.shape == x.shape


    @cartesian
    def source(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = 2*pi*pi*np.cos(pi*x)*np.cos(pi*y)
        return val#-self.solution(p)

    @cartesian
    def gradient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val # val.shape == p.shape

    @cartesian
    def flux(self, p):
        """
        @brief 真解通量
        """
        return -self.gradient(p)

    @cartesian
    def dirichlet(self, p):
        """
        @brief Dirichlet 边界条件 
        """
        return self.solution(p)

    @cartesian
    def is_dirichlet_boundary(self, p):
        """
        @brief Dirichlet 边界的判断函数
        """
        y = p[..., 1]
        return (np.abs(y - self.y1) < 1e-12) | (np.abs( y -  self.y0) < 1e-12)

    @cartesian
    def neumann(self, p, n):
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1) # (NQ, NE)
        return val

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        return np.abs(x - self.x1) < 1e-12

    @cartesian
    def robin(self, p, n):
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        val += self.kappa*self.solution(p) 
        return val

    @cartesian
    def is_robin_boundary(self, p):
        x = p[..., 0]
        return np.abs(x - self.x0) < 1e-12
        

def GenerateMat(nx,ny, mat_type=None, mat_path=None, need_rhs=False, 
                mesh_type='quad', space_p=1, kappa=1.0):
    x0 = 0.0
    x1 = 1.0
    y0 = 0.0
    y1 = 1.0
    pde = PDE(x0,x1,y0,y1,kappa)
    domain = pde.domain()

    if mesh_type == 'tri':
        mesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)
    elif mesh_type == 'quad':
        mesh = QuadrangleMesh.from_box(domain, nx=nx, ny=ny)

    space = LagrangeFESpace(mesh, p=space_p)
    bform = BilinearForm(space)
    # (\nabla u, \nabla v)
    bform.add_domain_integrator(ScalarLaplaceIntegrator(q=space_p+2)) 
    # <kappa u, v>
    rbi = ScalarRobinBoundaryIntegrator(pde.kappa,
            threshold=pde.is_robin_boundary, q=space_p+2)
    bform.add_boundary_integrator(rbi) 
    A = bform.assembly()

    lform = LinearForm(space)
    # (f, v)
    si = ScalarSourceIntegrator(pde.source, q=space_p+2)
    lform.add_domain_integrator(si)
    # <g_R, v> 
    rsi = ScalarRobinSourceIntegrator(pde.robin, threshold=pde.is_robin_boundary, q=space_p+2)
    lform.add_boundary_integrator(rsi)
    # <g_N, v>
    nsi = ScalarNeumannSourceIntegrator(pde.neumann, 
            threshold=pde.is_neumann_boundary, q=space_p+2)
    lform.add_boundary_integrator(nsi)
    #ipdb.set_trace()
    F = lform.assembly()

    # Dirichlet boundary condition
    bc = DirichletBC(space, 
            pde.dirichlet, threshold=pde.is_dirichlet_boundary) 
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
        self.DefineRandFloat('kappa',1.0,30.0)

        if self.para['space_p'] == 1:
            self.DefineRandInt('nx', 50, 200)
        elif self.para['space_p'] == 2:
            self.DefineRandInt('nx', 50, 110)
        elif self.para['space_p'] == 3:
            self.DefineRandInt('nx', 40, 70)

        self.CopyValue('nx', 'ny')

def test():
    print('mesh is quad, p=1')
    GenerateMat(200,200)

    print('mesh is quad, p=2')
    GenerateMat(110,110,space_p=2)

    print('mesh is quad, p=3')
    GenerateMat(70,70,space_p=3)


    print('mesh is tri, p=1')
    GenerateMat(200,200,mesh_type='tri')
    
    print('mesh is tri, p=2')
    GenerateMat(110,110,mesh_type='tri',space_p=2)
    
    print('mesh is tri, p=3')
    GenerateMat(70,70,mesh_type='tri',space_p=3)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', default='1', type=int, dest='nx')
    parser.add_argument('--ny', default='1', type=int, dest='ny')
    parser.add_argument('--mat_type', default=None, type=str, dest='mat_type')
    parser.add_argument('--mat_path', default=None, type=str, dest='mat_path')
    parser.add_argument('--need_rhs', default=False, type=bool, dest='need_rhs')
    parser.add_argument('--mesh_type', default='quad', type=str, dest='mesh_type')
    parser.add_argument('--space_p', default=1, type=int, dest='space_p')
    parser.add_argument('--kappa', default=1.0, type=float, dest='kappa')

    args = parser.parse_args()
    nx = args.nx
    ny = args.ny
    mat_type = args.mat_type
    mat_path = args.mat_path
    need_rhs = args.need_rhs
    mesh_type = args.mesh_type
    space_p = args.space_p
    kappa = args.kappa
    GenerateMat(nx,ny,mat_type,mat_path,need_rhs,mesh_type,space_p,kappa)
