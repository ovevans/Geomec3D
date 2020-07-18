import sys
sys.path.insert(0, '../')

from dolfin import split, plot, Constant
import matplotlib.pyplot as plt

from libs.grids import *
from libs.spaces import *
from libs.io import *
from libs.properties import *
from libs.bcs import *
from libs.linearsystem import *
from libs.solvers import *


grid = RectangleGrid(0., 1., 0., 1., 1, 1) # x0, x1, y0, y1, z0, z1, nx, ny, nz

space = DisplacementPressureSpace(grid, 1, 1) # u_degree, p_degree
(u, p) = space.TrialFunction()
(w, q) = space.TestFunction()

properties = getJsonData("../data/poroelastic_properties.json")
properties = PoroelasticProperties(properties["HardSediment"]) # Porous medium name

bcs = []
bcs = BoundaryConditions(grid, space)
bcs.addPressureHomogeneousBC(4)
bcs.addDisplacementHomogeneousBC(1, 0)
bcs.addDisplacementHomogeneousBC(2, 0)
bcs.addDisplacementHomogeneousBC(3, 1)
bcs.blockInitialize()

ls = LinearSystem()
stiffnessBlock = ls.stiffnessBlock(properties, u, w)
porePressureBlock = ls.porePressureBlock(properties, p, w)
solidVelocityBlock = ls.solidVelocityBlock(properties, u, q)
storageBlock = ls.storageBlock(properties, p, q)
fluidFlowBlock = ls.fluidFlowBlock(properties, p, q)
A = [[stiffnessBlock, 		porePressureBlock],
	 [solidVelocityBlock,	storageBlock + fluidFlowBlock]]
# A = ls.assembly(A, bcs.dirichlet)

forceVector = ls.forceVector(Constant((0., -10.e3)), w, 4)
b = [forceVector,
	 0]
# b = ls.assembly(b, bcs.dirichlet)

# X = space.Function()
# solver = Mumps(A, X, b)
# (u_h, p_h) = solver.solution.block_split()
# plt.figure(1)
# plot(u_h, title="Displacement")
# plt.figure(2)
# plot(p_h, title="Pressure")
# plt.show()