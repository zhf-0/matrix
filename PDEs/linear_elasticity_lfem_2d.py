import argparse
import numpy as np

from fealpy.mesh import TriangleMesh, QuadrangleMesh
from fealpy.functionspace import LagrangeFESpace as Space
from fealpy.fem import LinearElasticityOperatorIntegrator
from fealpy.fem import VectorSourceIntegrator
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.fem import DirichletBC
from fealpy.fem import VectorNeumannBCIntegrator 
from fealpy.decorator import cartesian

from Parameters import Parameter
from Utility import WriteMatAndVec

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
        return val

    @cartesian
    def dirichlet(self, p):
        val = np.array([0.0, 0.0], dtype=np.float64)
        return val

    @cartesian
    def neumann(self, p, n):
        val = np.array([-500, 0.0], dtype=np.float64)
        return val

    @cartesian
    def is_dirichlet_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = np.abs(x - self.x0) < 1e-13
        return flag

    @cartesian
    def is_neumann_boundary(self, p):
        x = p[..., 0]
        y = p[..., 1]
        flag = np.abs(x - self.x1) < 1e-13
        return flag


def GenerateMat(nx,ny, mat_type=None, mat_path=None, need_rhs=False, 
                mesh_type='quad', space_p=1, E=1e+5, nu=0.2):
    x0 = 0.0
    x1 = 1.0
    y0 = 0.0
    y1 = 1.0
    pde = PDE(x0,x1,y0,y1,E,nu)
    domain = pde.domain()

    if mesh_type == 'tri':
        mesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)
    elif mesh_type == 'quad':
        mesh = QuadrangleMesh.from_box(domain, nx=nx, ny=ny)

    space = Space(mesh, p=space_p, doforder='sdofs')
    uh = space.function(dim=2)

    vspace = 2*(space, ) 
    bform = BilinearForm(vspace)
    bform.add_domain_integrator(LinearElasticityOperatorIntegrator(pde.lam, pde.mu))
    bform.assembly()

    # source item 
    lform = LinearForm(vspace)
    lform.add_domain_integrator(VectorSourceIntegrator(pde.source, q=1))
    if hasattr(pde, 'neumann'):
        bi = VectorNeumannBCIntegrator(pde.neumann, threshold=pde.is_neumann_boundary, q=1)
        lform.add_boundary_integrator(bi)
    lform.assembly()

    A = bform.get_matrix()
    F = lform.get_vector()

    bc = DirichletBC(vspace, pde.dirichlet, threshold=pde.is_dirichlet_boundary)
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
        self.DefineRandFloat('E',1e4,5e5)
        self.DefineRandFloat('nu',0.01,0.5)

        if self.para['space_p'] == 1:
            self.DefineRandInt('nx', 80, 160)
        elif self.para['space_p'] == 2:
            self.DefineRandInt('nx', 40, 80)
        elif self.para['space_p'] == 3:
            self.DefineRandInt('nx', 30, 50)

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
    parser.add_argument('--E', default=1e+5, type=float, dest='E')
    parser.add_argument('--nu', default=0.2, type=float, dest='nu')

    args = parser.parse_args()
    nx = args.nx
    ny = args.ny
    mat_type = args.mat_type
    mat_path = args.mat_path
    need_rhs = args.need_rhs
    mesh_type = args.mesh_type
    space_p = args.space_p
    E = args.E
    nu = args.nu
    GenerateMat(nx,ny,mat_type,mat_path,need_rhs,mesh_type,space_p,E,nu)
