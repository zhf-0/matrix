import argparse
import numpy as np
import sympy as sym
from sympy.vector import CoordSys3D, curl
from scipy.sparse import spdiags

from fealpy.mesh import TetrahedronMesh
from fealpy.functionspace import FirstNedelecFiniteElementSpace3d 
from fealpy.fem import DirichletBC # 处理边界条件
from fealpy.decorator import cartesian

from Parameters import Parameter
from Utility import WriteMatAndVec


class PDE:
    def __init__(self,x0,x1,y0,y1,z0,z1,beta=1):
        """
             curl curl E - beta E = J     Omega
                       n \times E = g     Gamma
        """
        C = CoordSys3D('C')
        x = sym.symbols("x")
        y = sym.symbols("y")
        z = sym.symbols("z")

        f = sym.sin(sym.pi*C.z)*C.i + sym.sin(sym.pi*C.x)*C.j + sym.sin(sym.pi*C.y)*C.k 

        # f
        fx = f.dot(C.i).subs({C.x:x, C.y:y, C.z:z})
        fy = f.dot(C.j).subs({C.x:x, C.y:y, C.z:z})
        fz = f.dot(C.k).subs({C.x:x, C.y:y, C.z:z})

        self.Fx = sym.lambdify(('x', 'y', 'z'), fx, "numpy")
        self.Fy = sym.lambdify(('x', 'y', 'z'), fy, "numpy")
        self.Fz = sym.lambdify(('x', 'y', 'z'), fz, "numpy")

        # curl(f)
        cf = curl(f)
        cfx = cf.dot(C.i).subs({C.x:x, C.y:y, C.z:z})
        cfy = cf.dot(C.j).subs({C.x:x, C.y:y, C.z:z})
        cfz = cf.dot(C.k).subs({C.x:x, C.y:y, C.z:z})
        self.curlFx = sym.lambdify(('x', 'y', 'z'), cfx, "numpy")
        self.curlFy = sym.lambdify(('x', 'y', 'z'), cfy, "numpy")
        self.curlFz = sym.lambdify(('x', 'y', 'z'), cfz, "numpy")

        # curl(curl(f))
        ccf = curl(cf)
        ccfx = ccf.dot(C.i).subs({C.x:x, C.y:y, C.z:z})
        ccfy = ccf.dot(C.j).subs({C.x:x, C.y:y, C.z:z})
        ccfz = ccf.dot(C.k).subs({C.x:x, C.y:y, C.z:z})
        self.curlcurlFx = sym.lambdify(('x', 'y', 'z'), ccfx, "numpy")
        self.curlcurlFy = sym.lambdify(('x', 'y', 'z'), ccfy, "numpy")
        self.curlcurlFz = sym.lambdify(('x', 'y', 'z'), ccfz, "numpy")

        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1
        self.beta = beta

    def domain(self):
        return np.array([self.x0,self.x1,self.y0,self.y1,self.z0,self.z1])

    def solution(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        Fx = self.Fx(x, y, z)
        Fy = self.Fy(x, y, z)
        Fz = self.Fz(x, y, z)
        if type(Fx) is not np.ndarray:
            Fx = np.ones(x.shape, dtype=np.float_)*Fx
        if type(Fy) is not np.ndarray:
            Fy = np.ones(x.shape, dtype=np.float_)*Fy
        if type(Fz) is not np.ndarray:
            Fz = np.ones(x.shape, dtype=np.float_)*Fz
        f = np.c_[Fx, Fy, Fz] 
        return f 

    @cartesian
    def curl_solution(self, p):
        """!
        @param p: (..., N, ldof, 3)
        """
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        cFx = self.curlFx(x, y, z)
        cFy = self.curlFy(x, y, z)
        cFz = self.curlFz(x, y, z)
        if type(cFx) is not np.ndarray:
            cFx = np.ones(x.shape, dtype=np.float_)*cFx
        if type(cFy) is not np.ndarray:
            cFy = np.ones(x.shape, dtype=np.float_)*cFy
        if type(cFz) is not np.ndarray:
            cFz = np.ones(x.shape, dtype=np.float_)*cFz
        cf = np.c_[cFx, cFy, cFz] #(..., NC, ldof, 3)
        return cf 

    @cartesian
    def source(self, p):
        x = p[..., 0, None]
        y = p[..., 1, None]
        z = p[..., 2, None]
        ccFx = self.curlcurlFx(x, y, z)
        ccFy = self.curlcurlFy(x, y, z)
        ccFz = self.curlcurlFz(x, y, z)
        if type(ccFx) is not np.ndarray:
            ccFx = np.ones(x.shape, dtype=np.float_)*ccFx
        if type(ccFy) is not np.ndarray:
            ccFy = np.ones(x.shape, dtype=np.float_)*ccFy
        if type(ccFz) is not np.ndarray:
            ccFz = np.ones(x.shape, dtype=np.float_)*ccFz
        ccf = np.c_[ccFx, ccFy, ccFz] 
        return ccf - self.beta*self.solution(p)

    @cartesian
    def dirichlet(self, p, n):
        val = self.solution(p)
        return val

def GenerateMat(nx,ny,nz, mat_type=None, mat_path=None, need_rhs=False, 
                beta=1.0):
    x0 = 0.0
    x1 = 0.5
    y0 = 0.0
    y1 = 0.5
    z0 = 0.0
    z1 = 0.5
    pde = PDE(x0,x1,y0,y1,z0,z1,beta)
    domain = pde.domain()
    mesh = TetrahedronMesh.from_box(domain, nx=nx, ny=ny, nz=nz)
    space = FirstNedelecFiniteElementSpace3d(mesh)

    M = space.mass_matrix()
    A = space.curl_matrix()
    b = space.source_vector(pde.source)
    A = A - pde.beta * M 

    # Dirichlet boundary condition 
    Eh=space.function()
    isDDof = space.set_dirichlet_bc(pde.dirichlet, Eh)
    b = b - A@Eh
    b[isDDof] = Eh[isDDof]
    bdIdx = np.zeros(A.shape[0], dtype=np.int_)
    bdIdx[isDDof] = 1
    Tbd = spdiags(bdIdx, 0, A.shape[0], A.shape[0])
    T = spdiags(1-bdIdx, 0, A.shape[0], A.shape[0])
    A = T@A@T + Tbd

    # eliminate zeros in the matrix 
    eps = 10**(-15)
    A.data[ np.abs(A.data) < eps ] = 0
    A.eliminate_zeros()
    
    # write matrix A and rhs vector F  
    WriteMatAndVec(A,b,mat_type,mat_path,need_rhs)

class Para(Parameter):
    def __init__(self):
        super().__init__()
        
    def AddParas(self):
        self.DefineRandFloat('beta',1.0,100.0)

        self.DefineRandInt('nx', 10, 60)
        self.CopyValue('nx', 'ny')
        self.CopyValue('nx', 'nz')

def test():
    GenerateMat(10,10,10)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nx', default='1', type=int, dest='nx')
    parser.add_argument('--ny', default='1', type=int, dest='ny')
    parser.add_argument('--nz', default='1', type=int, dest='nz')
    parser.add_argument('--mat_type', default=None, type=str, dest='mat_type')
    parser.add_argument('--mat_path', default=None, type=str, dest='mat_path')
    parser.add_argument('--need_rhs', default=False, type=bool, dest='need_rhs')
    parser.add_argument('--beta', default=1.0, type=float, dest='beta')

    args = parser.parse_args()
    nx = args.nx
    ny = args.ny
    nz = args.nz
    mat_type = args.mat_type
    mat_path = args.mat_path
    need_rhs = args.need_rhs
    beta = args.beta
    GenerateMat(nx,ny,nz,mat_type,mat_path,need_rhs,beta)
