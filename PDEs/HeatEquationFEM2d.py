#!/usr/bin/env python3
import numpy as np
from fealpy.decorator import cartesian
from fealpy.mesh import MeshFactory as MF
from fealpy.timeintegratoralg import UniformTimeLine
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.boundarycondition import DirichletBC 
from fealpy.tools.show import showmultirate


class PDE:
    """

    u_t - c*\Delta u = f

    c = 1/16
    u(x, y, t) = sin(2*PI*x)*sin(2*PI*y)*exp(-t)

    domain = [0, 1]^2


    """
    def __init__(self,x0,x1,y0,y1):
        self.diffusionCoefficient = 1/16
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

    def domain(self):
        return [self.x0, self.x1, self.y0, self.y1]

    def init_value(self, p):
        return self.solution(p, 0.0)

    def solution(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        u = np.sin(2*pi*x)*np.sin(2*pi*y)*np.exp(-t)
        return u
    
    def diffusion_coefficient(self, p):
        return self.diffusionCoefficient

    def source(self, p, t):
        x = p[..., 0]
        y = p[..., 1]
        pi = np.pi
        k = self.diffusionCoefficient
        rhs = (-1+k*8*pi**2)*np.sin(2*pi*x)*np.sin(2*pi*y)*np.exp(-t)
        return rhs

    def dirichlet(self, p, t):
        return self.solution(p,t)

    def is_dirichlet_boundary(self, p):
        eps = 1e-14 
        return (p[..., 0] < eps) | (p[..., 1] < eps) | (p[..., 0] > 1.0 - eps) | (p[..., 1] > 1.0 - eps)


def GenerateMat(nx,ny):
    #=======================================
    # adjustable config parameters
    degree = 1

    x0 = 0.0
    x1 = 1.0
    y0 = 0.0
    y1 = 1.0

    t_start = 0.0
    t_end = 1.0
    nt = 1
    #=======================================

    pde = PDE(x0,x1,y0,y1)
    domain = pde.domain()
    smesh = MF.boxmesh2d(domain, nx=nx, ny=ny, meshtype='tri')
    tmesh = UniformTimeLine(t_start, t_end, nt) 

    space = LagrangeFiniteElementSpace(smesh, p=degree)

    c = pde.diffusionCoefficient
    A = c*space.stiff_matrix() 
    M = space.mass_matrix() 
    dt = tmesh.current_time_step_length() 
    G = M + dt*A 

    # initial condition
    uh0 = space.interpolation(pde.init_value)

    # next time step t1
    t1 = tmesh.next_time_level()
    print("t1=", t1)


    # right hand side vector when t1
    @cartesian
    def source(p):
        return pde.source(p, t1)
    F = space.source_vector(source)
    F *= dt
    F += M@uh0

    # Dirichlet boundary condition when t1
    @cartesian
    def dirichlet(p):
        return pde.dirichlet(p, t1)
    bc = DirichletBC(space, dirichlet)

    # generate matrix and right hand side vector when t1
    uh1 = space.function()
    GD, F = bc.apply(G, F, uh1)

    return GD
    
    # # solve linear system 
    # uh1[:] = spsolve(GD, F).reshape(-1)
    # uh0[:] = uh1

    # # next time step 
    # tmesh.advance()

if __name__ == '__main__':
    a = GenerateMat(512,512)


