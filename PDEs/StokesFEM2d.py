import numpy as np

from scipy.sparse import spdiags, bmat

from scipy.sparse import csr_matrix,hstack,vstack,spdiags,bmat
import matplotlib.pyplot as plt
from fealpy.mesh import MeshFactory as MF
from fealpy.functionspace import LagrangeFiniteElementSpace
from fealpy.functionspace import ScaledMonomialSpace2d 
from fealpy.boundarycondition import DirichletBC 
# from fealpy.pde.stokes_model_2d import StokesModelData_6 as PDE

from fealpy.decorator import cartesian


# class StokesModelData_5:
class PDE:
    def __init__(self,x0,x1,y0,y1, nu=1):
        self.nu = 1
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

    def domain(self):
        return [self.x0, self.x1, self.y0, self.y1]

    # def init_mesh(self, n=1, meshtype='tri'):
    #     node = np.array([
    #         (0, 0),
    #         (1, 0),
    #         (1, 1),
    #         (0, 1)], dtype=np.float)

    #     if meshtype == 'tri':
    #         cell = np.array([
    #             (1, 2, 0),
    #             (3, 0, 2)], dtype=np.int)
    #         mesh = TriangleMesh(node, cell)
    #         mesh.uniform_refine(n)
    #         return mesh
    #     elif meshtype == 'quad':
    #         nx = 2
    #         ny = 2
    #         mesh = StructureQuadMesh(self.box, nx, ny)
    #         mesh.uniform_refine(n)
    #         return mesh
    #     elif meshtype == 'poly':
    #         cell = np.array([
    #             (1, 2, 0),
    #             (3, 0, 2)], dtype=np.int)
    #         mesh = TriangleMesh(node, cell)
    #         mesh.uniform_refine(n)
    #         nmesh = TriangleMeshWithInfinityNode(mesh)
    #         pnode, pcell, pcellLocation = nmesh.to_polygonmesh()
    #         pmesh = PolygonMesh(pnode, pcell, pcellLocation)
    #         return pmesh

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

def GenerateMat(nx,ny):
    #=======================================
    # adjustable config parameters
    degree = 2 # must >= 2

    x0 = 0.0
    x1 = 1.0
    y0 = 0.0
    y1 = 1.0
    #=======================================
    
    pde = PDE(x0,x1,y0,y1)
    domain = pde.domain()
    mesh = MF.boxmesh2d(domain, nx=nx, ny=ny, meshtype='tri')

    uspace = LagrangeFiniteElementSpace(mesh, p=degree)
    pspace = LagrangeFiniteElementSpace(mesh, p=degree-1)

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

    return AA
    # ctx.set_centralized_sparse(AA)
    # xx = FF.copy()
    # ctx.set_rhs(xx)
    # ctx.run(job=6)
    # uh[:, 0] = xx[:ugdof]
    # uh[:, 1] = xx[ugdof:2*ugdof]
    # ph[:] = xx[2*ugdof:]

    # NDof[i] =  gdof 
    
    # uc1 = pde.velocity(mesh.node)
    # NN = mesh.number_of_nodes()
    # uc2 = uh[:NN]
    # up1 = pde.pressure(mesh.node)
    # up2 = ph[:NN]
    
    # NDof[i] =  gdof 
    # area = sum(mesh.entity_measure('cell'))

    # iph = pspace.integralalg.integral(ph)/area
    
    # ph[:] = ph[:]-iph
 
    # errorMatrix[0, i] = uspace.integralalg.error(pde.velocity, uh)
    # errorMatrix[1, i] = pspace.integralalg.error(pde.pressure, ph)
    # #errorMatrix[0, i] = np.abs(uc1-uc2).max()
    # #errorMatrix[1, i] = np.abs(up1-up2).max()
    # if i < maxit-1:
    #     mesh.uniform_refine()
        
    # ctx.destroy()
    # print(errorMatrix)
    # showmultirate(plt, 0, NDof, errorMatrix, errorType)
    # plt.show()

if __name__ == '__main__':
    a = GenerateMat(64,64)

