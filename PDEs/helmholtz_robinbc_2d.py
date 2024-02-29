import argparse
from scipy.special import jv
import numpy as np

from fealpy.mesh import TriangleMesh, QuadrangleMesh
from fealpy.functionspace import LagrangeFESpace
from fealpy.fem import ScalarDiffusionIntegrator      # (A\nabla u, \nabla v)
from fealpy.fem import ScalarMassIntegrator           # (r*u, v)
from fealpy.fem import ScalarSourceIntegrator         # (f, v)
from fealpy.fem import ScalarRobinSourceIntegrator    # <g_R, v>
from fealpy.fem import ScalarRobinBoundaryIntegrator  # <kappa*u, v>
from fealpy.fem import BilinearForm
from fealpy.fem import LinearForm
from fealpy.decorator import cartesian

from Parameters import Parameter
from Utility import WriteMatAndVec

class PDE():
    def __init__(self,x0,x1,y0,y1,k=1):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

        # k is the wave number
        self.k = k

    def domain(self):
        return np.array([self.x0, self.x1,self.y0, self.y1])

    @cartesian
    def solution(self, p):
        k = self.k
        x = p[..., 0]
        y = p[..., 1]
        r = np.sqrt(x**2 + y**2)

        val = np.zeros(x.shape, dtype=np.complex128)
        val[:] = np.cos(k*r)/k
        c = complex(np.cos(k), np.sin(k))/complex(jv(0, k), jv(1, k))/k
        val -= c*jv(0, k*r)
        return val

    @cartesian
    def gradient(self, p):
        """
        x*(I*sin(k) + cos(k))*besselj(1, R*k)/(R*(besselj(0, k) + I*besselj(1, k))) - x*sin(R*k)/R
        y*(I*sin(k) + cos(k))*besselj(1, R*k)/(R*(besselj(0, k) + I*besselj(1, k))) - y*sin(R*k)/R
        """
        k = self.k
        x = p[..., 0]
        y = p[..., 1]
        r = np.sqrt(x**2 + y**2)

        val = np.zeros(p.shape, dtype=np.complex128)
        t0 = np.sin(k*r)/r
        c = complex(np.cos(k), np.sin(k))/complex(jv(0, k), jv(1, k))
        t1 = c*jv(1, k*r)/r
        t2 = t1 - t0
        val[..., 0] = t2*x
        val[..., 1] = t2*y
        return val

    @cartesian
    def source(self, p):
        k = self.k
        x = p[..., 0]
        y = p[..., 1]
        r = np.sqrt(x**2 + y**2)
        val = np.zeros(x.shape, dtype=np.complex128)
        val[:] = np.sin(k*r)/r
        return val

    @cartesian
    def robin(self, p, n):
        k = self.k
        x = p[..., 0]
        y = p[..., 1]
        grad = self.gradient(p) # (NQ, NE, 2)
        val = np.sum(grad*n, axis=-1)
        kappa = np.broadcast_to(np.complex_(0.0 + 1j * k), shape=x.shape)
        val += kappa*self.solution(p) 
        return val


    def symbolic_com(self):
        import sympy as sp
        x, y, k , R= sp.symbols('x, y, k, R', real=True)
        r = sp.sqrt(x**2 + y**2)
        J0k = sp.besselj(0, k)
        J1k = sp.besselj(1, k)
        J0kr = sp.besselj(0, k*r)
        u = sp.cos(k*r)/k - J0kr*(sp.cos(k) + sp.I*sp.sin(k))/(J0k + sp.I*J1k)/k
        f = (-u.diff(x, 2) - u.diff(y, 2) - k**2*u).simplify().subs({r:R})
    
        print("f:", f)
        print(u.diff(x).subs({r:R}))
        print(u.diff(y).subs({r:R}))


def GenerateMat(nx,ny, mat_type=None, mat_path=None, need_rhs=False, 
                mesh_type='quad', space_p=1, k=1):
    x0 = 0.0
    x1 = 1.0
    y0 = 0.0
    y1 = 1.0
    pde = PDE(x0,x1,y0,y1,k)
    domain = pde.domain()
    kappa = k * 1j

    D = ScalarDiffusionIntegrator(c=1, q=space_p+2)
    M = ScalarMassIntegrator(c=-k**2, q=space_p+2)
    R = ScalarRobinBoundaryIntegrator(kappa=kappa, q=space_p+2)
    f = ScalarSourceIntegrator(pde.source, q=space_p+2)

    Vr = ScalarRobinSourceIntegrator(pde.robin, q=space_p+2)

    if mesh_type == 'tri':
        mesh = TriangleMesh.from_box(domain, nx=nx, ny=ny)
    elif mesh_type == 'quad':
        mesh = QuadrangleMesh.from_box(domain, nx=nx, ny=ny)

    mesh.ftype = complex
    space = LagrangeFESpace(mesh, p=space_p)

    b = BilinearForm(space)
    b.add_domain_integrator([D, M])
    b.add_boundary_integrator(R)

    l = LinearForm(space)
    l.add_domain_integrator(f)
    l.add_boundary_integrator([Vr])

    A = b.assembly() 
    F = l.assembly()

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
        self.DefineRandInt('k',1,30)

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
    parser.add_argument('--k', default=1, type=int, dest='k')

    args = parser.parse_args()
    nx = args.nx
    ny = args.ny
    mat_type = args.mat_type
    mat_path = args.mat_path
    need_rhs = args.need_rhs
    mesh_type = args.mesh_type
    space_p = args.space_p
    k = args.k
    GenerateMat(nx,ny,mat_type,mat_path,need_rhs,mesh_type,space_p,k)
