from fealpy.decorator import cartesian, barycentric
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import  ParametricLagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC
from fealpy.tools.show import showmultirate, show_error_table
import numpy as np

class PDE:
    def __init__(self,x0,x1,y0,y1,blockx,blocky):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.xstep = (x1-x0)/blockx 
        self.ystep = (y1-y0)/blocky
        self.coef1 = 10**np.random.uniform(0.0,5.0,(blocky+1,blockx+1))
        self.coef2 = 10**np.random.uniform(0.0,5.0,(blocky+1,blockx+1))

    def domain(self):
        return np.array([self.x0, self.x1,self.y0, self.y1])
    
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
        pi = np.pi
        val = np.cos(pi*x)*np.cos(pi*y)
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
        pi = np.pi
        val = 12*pi*pi*np.cos(pi*x)*np.cos(pi*y) 
        val += 2*pi*pi*np.sin(pi*x)*np.sin(pi*y) 
        val += np.cos(pi*x)*np.cos(pi*y)*(x**2 + y**2 + 1) 
        val -= pi*np.cos(pi*x)*np.sin(pi*y) 
        val -= pi*np.cos(pi*y)*np.sin(pi*x)
        return val

    @cartesian
    def gradient(self, p):
        """ 
		The gradient of the exact solution 
        """
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        val = np.zeros(p.shape, dtype=np.float64)
        val[..., 0] = -pi*np.sin(pi*x)*np.cos(pi*y)
        val[..., 1] = -pi*np.cos(pi*x)*np.sin(pi*y)
        return val # val.shape == p.shape

    @cartesian
    def diffusion_coefficient(self, p):
        x = p[..., 0]
        y = p[..., 1]
        xidx = x//self.xstep
        xidx = xidx.astype(np.int)
        yidx = y//self.ystep 
        yidx = yidx.astype(np.int)

        shape = p.shape+(2,)
        val = np.zeros(shape,dtype=np.float64)
        val[...,0,0] = self.coef1[xidx,yidx]
        # val[...,0,0] = 10.0
        val[...,0,1] = 1.0
        val[...,1,0] = 1.0
        val[...,1,1] = self.coef2[xidx,yidx]
        # val[...,1,1] = 2.0
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
        return self.solution(p)

def GenerateMat(nx,ny,blockx,blocky):
    pde = PDE(0,1,0,1,blockx,blocky)
    domain = pde.domain()
    mesh = MF.boxmesh2d(domain, nx=nx, ny=ny, meshtype='quad',p=1)

    # space = LagrangeFiniteElementSpace(mesh, p=1)
    space = ParametricLagrangeFiniteElementSpace(mesh, p=1)
    # NDof = space.number_of_global_dofs()
    uh = space.function() 	
    A = space.stiff_matrix(c=pde.diffusion_coefficient)
    # B = space.convection_matrix(c=pde.convection_coefficient)
    # M = space.mass_matrix(c=pde.reaction_coefficient)
    F = space.source_vector(pde.source)
    # A += B 
    # A += M
    
    bc = DirichletBC(space, pde.dirichlet)
    A, F = bc.apply(A, F, uh)

    return A

if __name__ == '__main__':
    a = GenerateMat(512,512,4,4)
