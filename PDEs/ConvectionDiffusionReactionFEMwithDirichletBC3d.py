from fealpy.decorator import cartesian, barycentric
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import  ParametricLagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate, show_error_table
import numpy as np


class PDE:
    def __init__(self,x0,x1,y0,y1,z0,z1,blockx,blocky,blockz):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.z0 = z0
        self.z1 = z1
        self.xstep = (x1-x0)/blockx 
        self.ystep = (y1-y0)/blocky
        self.zstep = (z1-z0)/blockz

        self.coef1 = 10**np.random.uniform(0.0,5.0,(blocky+1,blockx+1,blockz+1))
        self.coef2 = 10**np.random.uniform(0.0,5.0,(blocky+1,blockx+1,blockz+1))
        self.coef3 = 10**np.random.uniform(0.0,5.0,(blocky+1,blockx+1,blockz+1))

    def domain(self):
        return np.array([self.x0, self.x1,self.y0, self.y1, self.z0,self.z1])
    
    @cartesian
    def solution(self, p):
        """ 
		The exact solution 
        Parameters
        ---------
        p : 
        Examples
        -------
        p = np.array([0, 1], dtype=np.float64)
        p = np.array([[0, 1], [0.5, 0.5]], dtype=np.float64)
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)*np.cos(pi*z)
        return val # val.shape == x.shape

    @cartesian
    def source(self, p):
        """ 
		The right hand side of convection-diffusion-reaction equation
        INPUT:
            p: array object,  
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        val = 12*pi*pi*np.cos(pi*x)*np.cos(pi*y)*np.cos(pi*z) 
        val += 2*pi*pi*np.sin(pi*x)*np.sin(pi*y)*np.sin(pi*z) 
        val += np.cos(pi*x)*np.cos(pi*y)*np.cos(pi*z)*(x**2 + y**2 + z**2 + 1) 
        val -= pi*np.cos(pi*x)*np.sin(pi*y)*np.sin(pi*z) 
        val -= pi*np.cos(pi*y)*np.sin(pi*x)*np.sin(pi*z)
        return val

    @cartesian
    def gradient(self, p):
        """ 
		The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)*np.cos(pi*z)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)*np.sin(pi*z)
        val[..., 2] = -pi*np.sin(pi*x)*np.sin(pi*y)*np.cos(pi*z)
        return val # val.shape == p.shape

    @cartesian
    def diffusion_coefficient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        xidx = x//self.xstep
        xidx = xidx.astype(np.int)
        yidx = y//self.ystep 
        yidx = yidx.astype(np.int)
        zidx = z//self.zstep 
        zidx = zidx.astype(np.int)

        shape = p.shape+(3,)
        val = np.zeros(shape,dtype=np.float64)
        val[...,0,0] = self.coef1[xidx,yidx,zidx]
        val[...,0,1] = 1.0
        val[...,0,2] = 1.0

        val[...,1,0] = 1.0
        val[...,1,1] = self.coef2[xidx,yidx,zidx]
        val[...,1,2] = 1.0

        val[...,2,0] = 1.0
        val[...,2,1] = 1.0
        val[...,2,2] = self.coef3[xidx,yidx,zidx]

        return val

    @cartesian
    def convection_coefficient(self, p):
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    @cartesian
    def reaction_coefficient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]
        return  x**2 + y**2 + z**2

    @cartesian
    def dirichlet(self, p):
        return self.solution(p)

def GenerateMat(nx,ny,nz,blockx,blocky,blockz, meshtype='tet', space_p=1 ):
    pde = PDE(0,1,0,1,0,1,blockx,blocky,blockz)
    domain = pde.domain()
    
    mesh = MF.boxmesh3d(domain, nx=nx, ny=ny, nz=nz, meshtype=meshtype)
    if meshtype == 'tet':
        space = LagrangeFiniteElementSpace(mesh, p=space_p)
    elif meshtype == 'hex':
        # TODO
        # space = ParametricLagrangeFiniteElementSpace(mesh, p=space_p)
        pass

    uh = space.function() 	
    A = space.stiff_matrix(c=pde.diffusion_coefficient)
    B = space.convection_matrix(c=pde.convection_coefficient)
    M = space.mass_matrix(c=pde.reaction_coefficient)
    F = space.source_vector(pde.source)
    A += B 
    A += M
    
    bc = DirichletBC(space, pde.dirichlet)
    A, F = bc.apply(A, F, uh)

    return A

if __name__ == '__main__':
    print('mesh is tet, p=1')
    a = GenerateMat(10,10,10,2,2,2)
    print(a.shape,a.nnz)

    print('mesh is tet, p=2')
    a = GenerateMat(10,10,10,2,2,2,space_p=2)
    print(a.shape,a.nnz)

    print('mesh is tet, p=3')
    a = GenerateMat(10,10,10,2,2,2,space_p=3)
    print(a.shape,a.nnz)


    print('mesh is hex, p=1')
    a = GenerateMat(10,10,10,2,2,2,meshtype='hex')
    print(a.shape,a.nnz)
    
    print('mesh is hex, p=2')
    a = GenerateMat(10,10,10,2,2,2,meshtype='hex',space_p=2)
    print(a.shape,a.nnz)
    
    print('mesh is hex, p=3')
    a = GenerateMat(10,10,10,2,2,2,meshtype='hex',space_p=3)
    print(a.shape,a.nnz)
